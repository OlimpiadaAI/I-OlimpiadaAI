# Olimpiada Sztucznej Inteligencji

Witamy w repozytorium z zadaniami Olimpiady Sztucznej Inteligencji. Olimpiada ta jest skierowana do uczniów szkół średnich w Polsce, którzy są zainteresowani sztuczną inteligencją. Celem jest zwiększenie zainteresowania AI oraz wyłonienie drużyny na Międzynarodową Olimpiadę Sztucznej Inteligencji.

<p align="center">
  <img src="https://raw.githubusercontent.com/OlimpiadaAI/I-OlimpiadaAI/main/logo_ioai.png" width="40%">
</p>

## Informacje ogólne

**Strona główna:** [Olimpiada Sztucznej Inteligencji](https://oai.cs.uni.wroc.pl/polski)

Olimpiada jest dwuetapowa, z pierwszym etapem zdalnym trwającym od 22. kwietnia do 27. maja, w którym uczestnicy rozwiązują zadania w domu. Prosimy nie udostępniać rozwiązań przed zakończeniem konkursu. Regulamin konkursu jest dostępny na naszej stronie.

## Sposób oddawania zadań

Zadania powinny być rozwiązane samodzielnie i przesłane do Komitetu Zadaniowego za pomocą specjalnej strony Olimpiady. Każde zadanie określa, jakie pliki należy przesyłać – najczęściej będzie to jeden plik Jupyter Notebook. Wszystkie prace będą oceniane automatycznie przez skrypt podobny do zawartego w zadaniu. Przed wysłaniem rozwiązania **każdy uczestnik powinien upewnić się, że działa ono na skrypcie walidacyjnym**.

## Zadania

W ramach Olimpiady uczestnicy zmierzą się z następującymi wyzwaniami:
- **Ataki adwersarialne** – Atak na konwolucyjną sieć neuronową.
- **Niezbalansowana klasyfikacja** – Trening klasyfikatora na niezbalansowanych danych.
- **Analiza zaleznosciowa** – Analiza składniowa zdań przy użyciu modelu HerBERT.
- **Kwantyzacja kolorów** – Kwantyzacja kolorów w obrazach.
- **Śledzenie obiektow** – Śledzenie obiektów w sekwencji wideo.
- **Pruning** – Zmniejszanie liczby wag w sieciach neuronowych.
- **Zagadki** – Odpowiadanie na pytania do tekstu źródłowego.

## Regulamin i kryteria oceny

Lista dopuszczalnych pakietów znajduje się w pliku `requirements.txt`. Rozwiązania będą testowane przy użyciu Pythona 3.11. Na potrzeby pracy nad zadaniami, zalecamy stworzenie środowiska wirtualnego
```
python3 -m venv oai_env
source oai_env/bin/activate
pip install -r OlimpiadaAI/requirements.txt
```

## Licencje

Repozytorium korzysta z następujących zasobów objętych licencjami:

- **Składnica zależnościowa** - Zasób dostępny na licencji GNU General Public License wersja 3 (GPLv3). Więcej informacji można znaleźć [tutaj](https://zil.ipipan.waw.pl/Sk%C5%82adnica). Zbiór danych użyty w zadaniu `syntax_trees` stanowi utwór pochodny.
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

## Kontakt

W razie pytań lub wątpliwości, prosimy o kontakt przez e-mail: [oai@cs.uni.wroc.pl](mailto:oai@cs.uni.wroc.pl).

Życzymy inspiracji i powodzenia w rozwiązywaniu zadań!
