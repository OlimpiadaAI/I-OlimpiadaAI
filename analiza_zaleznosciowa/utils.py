import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Callable, List
import nltk
from nltk.tree import Tree
import numpy as np
from torch.utils.data import Dataset


class Sentence:
    def __init__(self, words):
        self.words = words
    
    def __len__(self):
        return len(self.words)

    def __repr__(self):
        return ' '.join(self.words)


class ParsedSentence(Sentence):
    def __init__(self, words, heads):
        if len(words) != len(heads):
            raise ValueError("Different number of words and heads")
        if not all(0 <= head <= len(words) for head in heads):
            raise ValueError("Invalid head index")
        if heads.count(0) != 1:
            raise ValueError("There should be exactly one root node")
        super().__init__(words)
        self.words = words
        self.heads = heads
        self.root, self.node_to_children = self.get_root_and_children_list()

    @staticmethod
    def from_edges_and_root(word_tokens, edges, root):
        adj = [[] for _ in word_tokens]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        heads = [0] * len(word_tokens)

        def helper(v, parent):
            heads[v] = parent + 1
            for u in adj[v]:
                if u != parent:
                    assert heads[u] == 0
                    helper(u, v)
        helper(root, -1)
        return ParsedSentence(word_tokens, heads)

    def __getitem__(self, idx):
        return self.words[idx], self.heads[idx]

    def get_sorted_edges(self):
        edges = []
        for i, head in enumerate(self.heads):
            if head != 0:
                edge = (min(i, head - 1), max(i, head - 1))
                edges.append(edge)
        return edges

    def get_root_and_children_list(self):
        root = None
        node_to_children = [[] for _ in self.words]
        for i, head in enumerate(self.heads):
            if head == 0:
                root = i
            else:
                node_to_children[head-1].append(i)
        if root is None:
            raise ValueError("No root found")
        return root, node_to_children

    def to_tree(self) -> Tree:
        root = self.root
        node_to_children = self.node_to_children
        def helper(idx: int, depth: int = 0) -> Tree:
            if len(node_to_children[idx]) == 0:
                return self.words[idx]
            if depth  > 13:
                raise ValueError("Tree too deep")
            
            chlds = [helper(child, depth + 1) for child in node_to_children[idx]]
            return Tree(self.words[idx], chlds)
        return helper(root)
    
    def pretty_print(self):
        self.to_tree().pretty_print()


def uuas_score(golden_sent: ParsedSentence, pred_sent: ParsedSentence):
    gold_edges = set(golden_sent.get_sorted_edges())
    pred_edges = set(pred_sent.get_sorted_edges())
    correct_edges = gold_edges & pred_edges
    return len(correct_edges) / len(gold_edges)


def read_conll(filepath) -> List[ParsedSentence]:
    """Wczytuje sparsowane zdania z pliku o rozszerzeniu conll."""
    sentences = []
    with open(filepath) as f:
        words, heads = [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) >= 2:
                words.append(sp[1])
                heads.append(int(sp[6]))
            elif len(words) > 0:
                sentences.append(ParsedSentence(words, heads))
                words, heads = [], []
        if len(words) > 0:
            sentences.append(ParsedSentence(words, heads))
    return sentences


def merge_subword_tokens(
    word_tokens: List[str],
    subword_tokens: List[torch.Tensor],
    subword_embeddings: List[torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    agg_fn: Callable[[List[np.ndarray]], np.ndarray],
    verbose: bool = False,
):
    """Dla danego zdania funkcja zwraca dwuwymiarowy tensor reprezentujący embeddingi słów.

    Drzewa zależności są budowane na tokenach słów, a tokenizatory LLM-ów zwykle operują na podsłowach.
    Na przykład słowo "kurzawie" jest podzielone na tokeny "ku", "rza" i "wie" przez tokenizer HERBERT-a.
    Dlatego musimy z embeddingów połączyć niektóre embeddingi tokenów podsłów, aby uzyskać embeddingi słów.

    Argumenty:
        word_tokens (List[str]): Reprezentacja wejściowego zdania poprzez listę jego słów.
        subword_tokens (List[torch.Tensor]): Lista uprzednio wyliczonych tokenów podsłów dla wejściowego zdania.
        subword_embeddings (List[torch.Tensor]): Lista uprzednio wyliczonych embeddingów podsłów dla wejściowego zdania.
        tokenizer (PreTrainedTokenizer): Tokenizator użyty do uzyskania tokenów podsłów (subword_tokens).
        agg_fn (Callable[[List[torch.Tensor]], torch.Tensor]): Funkcja agregująca embeddingi podsłów w embeddingi słów. 
            Funkcja agg_fn powinna przyjmować listę embeddingów podsłów i zwracać pojedynczy embedding słowa.
        verbose (bool, optional): Czy wypisać na standardowe wyjście tokeny podsłów i ich odpowiadające słowa. Domyślnie False.

    Zwraca:
        torch.Tensor: Dwuwymiarowy tensor reprezentujący embeddingi słów. Pierwszy wymiar odpowiada indeksom słów, 
            a drugi wymiar odpowiada wymiarom embeddingu.

    Wyjątki:
        ValueError: Wyrzucany, jeśli nie znaleziono dopasowania słów do tokenów podsłów.
    """
    i = 0
    result = []
    tokens_ = []
    for word_token in word_tokens:
        tokens_in_word = tokenizer.batch_encode_plus(
            [word_token],
            padding="longest",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        tokens_ += tokens_in_word

        if verbose:
            int_list = tokens_in_word.tolist()
            tokens_in_word = [tokenizer.decode([tok]) for tok in int_list]
            print(f"{word_token: <15} ->  {', '.join(tokens_in_word)}")

        embeddings = subword_embeddings[i: i + len(tokens_in_word)]
        i += len(tokens_in_word)

        result += [agg_fn(embeddings)]
    
    tokens_ = torch.tensor([t.item() for t in tokens_])
    if tokens_.shape != subword_tokens.shape or torch.any(tokens_ != subword_tokens):
        print(tokens_, subword_tokens)
        raise ValueError("Token mismatch")
    return torch.stack(result)


class ListDataset(Dataset):
    def __init__(self, examples: List):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
