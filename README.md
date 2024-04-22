# Olimpiada Sztucznej Inteligencji

Witamy w repozytorium z zadaniami Olimpiady Sztucznej Inteligencji. Olimpiada ta jest skierowana do uczniów szkół średnich w Polsce, którzy są zainteresowani sztuczną inteligencją. Celem jest zwiększenie zainteresowania AI oraz wyłonienie drużyny na Międzynarodową Olimpiadę Sztucznej Inteligencji.

<p align="center">
  <img src="https://github.com/OlimpiadaAI/OlimpiadaAI/assets/150927210/66d4b9dc-c761-41c4-b128-1c67c39d6b74" width="40%">
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
- **HerBERT base cased** - Model dostępny dostępny [tu](https://huggingface.co/allegro/herbert-base-cased), Mroczkowski, R., Rybak, P., Wróblewska, A., & Gawlik, I. (2021). *HerBERT: Efficiently pretrained transformer-based language model for Polish*. arXiv preprint arXiv:2105.01735.
- **Zbiory danych generowane przy użyciu PyBullet** - objęte licencją MIT, szczegóły [tutaj](https://github.com/hebaishi/pybullet/blob/master/LICENSE).
- **Dall-E i Stable Diffusion** - pełne prawa do użycia i sprzedaży wyników, więcej informacji w [licencji](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE).
- **Zbiory danych generowane przy użyciu SCGAN** - więcej informacji na [IEEE](https://ieeexplore.ieee.org/document/8476290) oraz w [repozytorium GitHub](https://github.com/gauss-clb/SCGAN).

## Kontakt

W razie pytań lub wątpliwości, prosimy o kontakt przez e-mail: [oai@cs.uni.wroc.pl](mailto:oai@cs.uni.wroc.pl).

Życzymy inspiracji i powodzenia w rozwiązywaniu zadań!
