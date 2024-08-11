# I Olimpiada Sztucznej Inteligencji / 1st Polish AI Olympiad

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/OlimpiadaAI/I-OlimpiadaAI/blob/master/README.md)

Witamy w repozytorium z zadaniami I Olimpiady Sztucznej Inteligencji. Olimpiada ta jest skierowana do uczniów szkół średnich w Polsce, którzy są zainteresowani sztuczną inteligencją. Celem jest zwiększenie zainteresowania AI oraz wyłonienie drużyny na [Międzynarodową Olimpiadę Sztucznej Inteligencji](https://ioai-official.org/).

<p align="center">
  <img src="https://raw.githubusercontent.com/OlimpiadaAI/I-OlimpiadaAI/main/logo_ioai.png" width="40%">
</p>

## Informacje ogólne

**Strona główna:** [Olimpiada Sztucznej Inteligencji](https://oai.cs.uni.wroc.pl/polski)

Pierwsza edycja Olimpiady odbyła się w dwóch etapach. W pierwszym etapie, który trwał od 22 kwietnia do 27 maja 2024, uczestnicy rozwiązywali zadania w domu. 30 najlepszych uczestników zostało zaproszonych do udziału w obozie finałowym odbywającym się od 15 do 21 czerwca 2024. W czasie obozu finałowego odbyły się dwa konkursy: finałowy oraz implementacyjny. Regulaminy dostępne są na naszej stronie.

## Sposób oddawania zadań

Uczestnicy rozwiązywali zadania samodzielnie i przesyłali je do Komitetu Zadaniowego za pomocą specjalnej strony Olimpiady. Każde zadanie określa, jakie pliki należało przesyłać – najczęściej jest to jeden plik Jupyter Notebook, a czasem dodatowo należało przesłać wagi wytrenowanego modelu. Wszystkie prace były oceniane automatycznie przez skrypt podobny do zawartego w zadaniu. Przed wysłaniem rozwiązania należało upewnić się, że działa ono na skrypcie walidacyjnym.

## Zadania

W ramach 1 etapu Olimpiady uczestnicy zmierzyli się z następującymi wyzwaniami:
- **Ataki adwersarialne** – Atak na konwolucyjną sieć neuronową.
- **Niezbalansowana klasyfikacja** – Trening klasyfikatora na niezbalansowanych danych.
- **Analiza zależnościowa** – Analiza składniowa zdań przy użyciu modelu HerBERT.
- **Kwantyzacja kolorów** – Kwantyzacja kolorów w obrazach.
- **Śledzenie obiektów** – Śledzenie obiektów w sekwencji wideo.
- **Pruning** – Zmniejszanie liczby wag w sieciach neuronowych.
- **Zagadki** – Odpowiadanie na pytania do tekstu źródłowego.

W ramach konkursu finałowego, uczestnicy rozwiązywali następujące zadania:
- **Szyfry** – Zaprojektowanie algorytmu uczenia maszynowego do znalezienia tekstu jawnego.
- **Detekcja anomalii** – Wykrywanie obrazów spoza próby.
- **Self-supervised learning** – Klasyfikacja szeregów czasowych.

W ramach konkursu implementacyjnego **Tłumaczenie maszynowe**, uczestnicy zmierzyli się z implementacją i uproszczoną reprodukcją pracy naukowej.

## Środowisko

Lista dopuszczalnych pakietów znajduje się w pliku `requirements.txt`. Rozwiązania były testowane przy użyciu Pythona 3.11. Na potrzeby pracy nad zadaniami, zalecamy stworzenie środowiska wirtualnego
```
python3 -m venv oai_env
source oai_env/bin/activate
pip install -r OlimpiadaAI/requirements.txt
```

## Kryteria oceny
Oceny za zadania zostały wyliczone na podstawie podanych w treściach zadań kryteriów. Za zadania będzie można zdobyć maksymalnie 1.0 (Ataki adwersarialne, Niezbalansowana klasyfikacja), 1.5 (Śledzenie obiektow, Pruning, Zagadki, Kwantyzacja kolorów) lub 2.0 punkty (Analiza zależnosciowa). Łącznie w pierwszym etapie jest do zdobycia 10 punktów. W konkursie finałowym, wszystkie zadania były równo punktowane.

## Licencje

Repozytorium korzysta z następujących zasobów objętych licencjami:

- **Składnica zależnościowa** - Zasób dostępny na licencji GNU General Public License wersja 3 (GPLv3). Więcej informacji można znaleźć [tutaj](https://zil.ipipan.waw.pl/Sk%C5%82adnica). Zbiór danych użyty w zadaniu "Analiza zależnościowa" stanowi utwór pochodny.
- **HerBERT base cased** - Model dostępny dostępny [tu](https://huggingface.co/allegro/herbert-base-cased),
```
@inproceedings{mroczkowski-etal-2021-herbert,
    title = "{H}er{BERT}: Efficiently Pretrained Transformer-based Language Model for {P}olish",
    author = "Mroczkowski, Robert  and
          Rybak, Piotr  and
          Wr{\\'o}blewska, Alina  and
          Gawlik, Ireneusz",
    booktitle = "Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing",
    month = apr,
    year = "2021",
    address = "Kiyv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.bsnlp-1.1",
    pages = "1--10",
}
```
- **Zbiory danych generowane przy użyciu PyBullet** - objęte licencją MIT, szczegóły [tutaj](https://github.com/hebaishi/pybullet/blob/master/LICENSE).
- **Dall-E i Stable Diffusion** - pełne prawa do użycia i sprzedaży wyników, więcej informacji w [licencji](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE).
- **Zbiory danych generowane przy użyciu SCGAN** - więcej informacji na [IEEE](https://ieeexplore.ieee.org/document/8476290) oraz w [repozytorium GitHub](https://github.com/gauss-clb/SCGAN).

- W ramach zawodów finałowych uczniowie implementowali pracę 
```
@article{Bahdanau2014NeuralMT,
  title={Neural Machine Translation by Jointly Learning to Align and Translate},
  author={Dzmitry Bahdanau and Kyunghyun Cho and Yoshua Bengio},
  journal={CoRR},
  year={2014},
  volume={abs/1409.0473},
  url={https://api.semanticscholar.org/CorpusID:11212020}
}
```

## Kontakt

W razie pytań lub wątpliwości, prosimy o kontakt przez e-mail: [oai@cs.uni.wroc.pl](mailto:oai@cs.uni.wroc.pl).

Życzymy inspiracji i powodzenia w rozwiązywaniu zadań!
