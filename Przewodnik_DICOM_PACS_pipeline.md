# End-to-end pipeline DICOM → PACS
### Przewodnik do samodzielnej budowy (ucz się przez robienie)

> **Kontrakt tego dokumentu.** To nie jest gotowiec do skopiowania. Każda faza mówi
> Ci **co zbudować, dlaczego to działa i po czym poznasz, że działa** — a kod pojawia
> się tylko jako szkielet z `TODO`. Jeśli w którymś miejscu masz ochotę „po prostu
> skopiować rozwiązanie”, to jest dokładnie ten moment, w którym najwięcej się nauczysz,
> pisząc go samodzielnie. Cel końcowy: wrzucasz badanie do PACS, a po kilkunastu
> sekundach w przeglądarce pojawia się nakładka segmentacji z Twojego modelu — plus
> dashboard z metrykami. Efekt: portfolio, którego nie ma żaden inny junior.

---

## Jak korzystać z tego przewodnika

- **Rób fazy po kolei.** Każda kończy się checkpointem. Nie przechodź dalej, dopóki
  checkpoint nie świeci na zielono — inaczej debugujesz pięć rzeczy naraz.
- **Prowadź dziennik (`JOURNAL.md`).** Po każdej sesji zapisz: co zrobiłem, co nie
  działało, jak naprawiłem. To jest 50% wartości edukacyjnej i gotowy materiał do README.
- **Commituj małe kawałki.** Jeden commit = jeden działający krok. To Twój dowód
  procesu, nie tylko efektu.
- **Nie znasz czegoś? Najpierw przeczytaj docs, potem pytaj.** Przy każdej fazie mam
  sekcję *Do przeczytania* — to nie ozdoba, to część zadania.

**Czego już nie muszę Ci tłumaczyć** (masz to z CV): Docker, docker-compose, obrazy
i sieci kontenerów, CI/CD w GitHub Actions, Grafana, PyTorch, MONAI, trening 3D U-Net.
**Czego się tu nauczysz** (nowe): model danych DICOM, czym jest PACS, protokoły DIMSE
vs DICOMweb, obiekty DICOM-SEG i DICOM-SR, jak model „rozmawia” z systemem szpitalnym.
W Części II dochodzi cykl życia modelu: wersjonowanie danych (DVC), tracking i rejestr
modeli (MLflow), ewaluacja, która wytrzyma pytania recenzenta, bramka jakości w CI,
niepewność modelu, optymalizacja inferencji (ONNX/TensorRT/Triton), drift i pętla
zwrotna od radiologa.

**Układ przewodnika:**
- **Część I (Fazy 0–6): pipeline kliniczny.** Badanie wchodzi do PACS, wynik modelu
  wraca jako DICOM-SEG/SR i wyświetla się w OHIF.
- **Część II (Fazy 7–11): cykl życia modelu, czyli ML i MLOps.** Skąd model się bierze,
  skąd wiesz, że jest dobry, jak bezpiecznie wymienić go na lepszy i jak zauważyć,
  że przestał działać.
- **Część III (Faza 12): portfolio.** README, demo, droga do produkcji.

**Realny budżet czasu:** Część I to 3–4 weekendy, jeśli robisz solo i uczciwie czytasz
docs. Faza 0–2 to jeden weekend. Faza 3 to serce projektu i zajmie najwięcej. Faza 6 gra
na Twoją siłę (DevOps), więc pójdzie szybko. Część II to kolejne 4–6 weekendów plus czas
GPU: trening baseline'u nnU-Net i eksperymenty active learning to dni obliczeń, więc
puszczaj je w tle i w tym czasie pisz kod następnej fazy.

**Mało czasu na Część II?** Minimum, które najbardziej zmienia odbiór projektu:
Faza 7 (splity + DVC + MLflow) → Faza 8 (ewaluacja z nnU-Net i zbiorem zewnętrznym)
→ Faza 9 (rejestr, bramka w CI, shadow) → Faza 11.2 (poprawki radiologa jako etykiety).
Faza 10 oraz 11.1, 11.3 i 11.4 to rozszerzenia.

---

# Część I — Pipeline kliniczny

## Faza 0 — Model mentalny (nie pomijaj, to fundament)

Zanim cokolwiek uruchomisz, musisz rozumieć **dlaczego obrazowanie medyczne nie jest
zwykłym CV**. To jest różnica między kandydatem, który „trenował U-Net”, a takim, który
„rozumie workflow kliniczny”.

### 0.1 Model danych DICOM — hierarchia

DICOM to nie „format pliku”, to model danych + protokół sieciowy. Zapamiętaj hierarchię,
bo będzie wracać wszędzie:

```
Patient  (pacjent, np. PatientID)
└── Study   (badanie, jedna wizyta — StudyInstanceUID)
    └── Series  (seria, jedna akwizycja/modalność — SeriesInstanceUID)
        └── Instance  (pojedynczy plik/obraz — SOPInstanceUID)
```

Każdy poziom ma globalnie unikalny **UID**. To kluczowe: kiedy Twój model wypluje
segmentację, obiekt DICOM-SEG **musi się odwołać** do SeriesInstanceUID oryginału —
inaczej PACS nie wie, do czego przypiąć nakładkę. To jedna z pierwszych rzeczy, które
łamią początkującym.

Plik DICOM = nagłówek (setki tagów: `(0010,0010) PatientName`, `(0020,000D)
StudyInstanceUID`, `(0028,0010) Rows`, …) + piksele. Tagi są zdefiniowane w standardzie
(NEMA). Modalności które Cię interesują: `MG` (mammografia), `MR`, `CT`, `US`.

**Zadanie 0.1.** Zainstaluj `dcmtk` (`apt install dcmtk`) i pobierz jedno publiczne
badanie (patrz Faza 1). Uruchom `dcmdump plik.dcm | less` i **znajdź ręcznie** pięć
tagów: PatientID, StudyInstanceUID, SeriesInstanceUID, Modality, PixelSpacing.
Zapisz w dzienniku, co znaczy PixelSpacing i dlaczego model musi je znać.

### 0.2 Czym jest PACS i dwa protokoły

**PACS** (Picture Archiving and Communication System) = szpitalny magazyn obrazów +
usługi sieciowe do ich wysyłania/odbierania/wyszukiwania. W tym projekcie rolę PACS
gra **Orthanc** — lekki, open-source, idealny do nauki.

Są **dwa światy komunikacji** i musisz rozumieć oba:

| | DIMSE (klasyczny DICOM) | DICOMweb (nowoczesny, REST) |
|---|---|---|
| Transport | TCP, port zwykle 4242 | HTTP(S), REST/JSON |
| Wysłanie obrazu | C-STORE | STOW-RS |
| Wyszukanie | C-FIND | QIDO-RS |
| Pobranie | C-MOVE / C-GET | WADO-RS |
| Tożsamość węzła | **AE Title** (AET) | URL + auth |
| Kto tego używa | aparaty, stare PACS-y | przeglądarki, chmura, Twój kod |

Aparat CT czy mammograf „gada” DIMSE. Przeglądarka (OHIF) i Twój Python „gadają”
DICOMweb. Orthanc mówi w obu i tłumaczy między nimi — dlatego jest świetny do nauki.

**Zadanie 0.2.** Wyjaśnij w dzienniku (własnymi słowami, 3 zdania), czym różni się
C-STORE od STOW-RS i dlaczego szpital nadal używa DIMSE, skoro DICOMweb jest wygodniejszy.

### 0.3 Trzy typy obiektów wyjściowych — to jest sedno

Twój model **nie wypluwa PNG-a z maską**. W świecie klinicznym wynik AI to też obiekt
DICOM, żeby wrócił do PACS i wyświetlił się w każdym viewerze. Trzy formaty, które
musisz znać:

- **DICOM-SEG** (Segmentation) — maska segmentacji jako obiekt DICOM, odwołujący się do
  serii źródłowej. To wypluje Twój model raka piersi. Nakładka pojawia się na obrazie.
- **DICOM-SR** (Structured Report) — raport z **wartościami**: objętość zmiany, liczba
  ognisk, pewność modelu, mierzone odległości. Ustrukturyzowany, maszynowo czytelny.
- **DICOM RT-STRUCT** — kontury (używane głównie w radioterapii). Wiedz, że istnieje;
  w tym projekcie go pominiemy.

**Dlaczego to ważne dla rekrutera:** kandydat, który zapisuje maskę jako `.png`,
sygnalizuje „projekt studencki”. Kandydat, który zapisuje **DICOM-SEG odwołujący się do
serii źródłowej + DICOM-SR z objętością zmiany**, sygnalizuje „rozumiem, że model to
element systemu, nie notebook”.

> **Do przeczytania (Faza 0):**
> - Orthanc Book, rozdział „DICOM basics” — orthanc.uclouvain.be/book
> - „DICOM is Easy” (blog Roni Zaharia) — najlżejsze wprowadzenie do tagów i UID-ów
> - Wikipedia: „DICOM” — sekcje Data format + Network protocol wystarczą

**✅ Checkpoint Fazy 0:** potrafisz narysować z pamięci hierarchię Patient→Study→
Series→Instance, wymienić parę C-STORE/STOW-RS i wyjaśnić, czym różni się DICOM-SEG
od zwykłej maski PNG. Jak tak — idź dalej.

---

## Faza 1 — Postaw Orthanc i przepuść przez niego dane

Cel: mieć działający „szpitalny magazyn”, do którego wrzucisz badanie i wyciągniesz je
przez REST.

### 1.1 Uruchom Orthanc z wtyczką DICOMweb

Nie klonuj cudzego gotowego repo „na start” — napisz własny `docker-compose.yml`. To
kilka linijek, a zrozumienie każdej z nich jest celem.

```yaml
# docker-compose.yml  (szkielet — uzupełnij TODO sam)
services:
  orthanc:
    image: orthancteam/orthanc:latest   # zawiera wtyczki DICOMweb i OHIF
    ports:
      - "8042:8042"     # TODO: co to za port? (REST + web UI)
      - "4242:4242"     # TODO: a ten? (DIMSE / C-STORE)
    volumes:
      - ./orthanc.json:/etc/orthanc/orthanc.json:ro
      - orthanc-db:/var/lib/orthanc/db
    # TODO: dodaj healthcheck na /system (nauczysz się API przy okazji)
volumes:
  orthanc-db:
```

Konfiguracja Orthanca to jeden plik JSON. Minimalny zestaw, który musisz **świadomie**
ustawić (poczytaj, co robi każdy klucz — nie kopiuj na ślepo):

```jsonc
// orthanc.json  (szkielet)
{
  "Name": "TWOJ-PACS-DEV",
  "DicomAet": "ORTHANC",          // ← to jest AE Title tego węzła (Faza 0.2!)
  "DicomServerEnabled": true,     // włącza DIMSE na 4242
  "DicomWeb": { "Enable": true, "Root": "/dicom-web/" },  // włącza QIDO/WADO/STOW
  "RemoteAccessAllowed": true,
  "AuthenticationEnabled": true,
  "RegisteredUsers": { "dev": "TODO-ustaw-haslo" }
  // TODO: doczytaj OrthancExplorer2 i włącz nowy UI — wygodniejszy
}
```

### 1.2 Zdobądź prawdziwe dane

Nie używaj danych szpitalnych do publicznego repo. Do nauki i do publicznej wersji
projektu bierz zbiory otwarte:

- **TCIA** (cancerimagingarchive.net) — natywnie DICOM, są kolekcje mammografii i MRI
  piersi. Idealne, bo pasują do tematu Twojej inżynierki.
- Dowolne przykładowe serie CT/MR z próbek `dcmtk` albo z zestawów MONAI (część jest
  w NIfTI — o konwersji za chwilę).

### 1.3 Wrzuć badanie i wyciągnij je przez REST

Dwie drogi wrzucenia — zrób **obie**, żeby poczuć różnicę DIMSE vs DICOMweb:

```bash
# Droga A — DIMSE (jak aparat): C-STORE narzędziem storescu z dcmtk
storescu -aec ORTHANC localhost 4242 sciezka/do/*.dcm
#   TODO: co znaczy -aec? (AE Title odbiorcy)

# Droga B — DICOMweb (jak przeglądarka): przez web UI Orthanca (drag & drop)
#   albo STOW-RS curl-em na /dicom-web/studies
```

Teraz **wyciągnij** dane wyłącznie przez API (żeby zrozumieć, jak potem zrobi to model):

```bash
# QIDO-RS: wylistuj badania
curl -u dev:haslo http://localhost:8042/dicom-web/studies

# Orthanc REST (inny, wygodny do skryptów): lista pacjentów, studies, series
curl -u dev:haslo http://localhost:8042/studies
```

**Zadanie 1.3.** Napisz krótki skrypt w Pythonie (`explore.py`), który przez REST
Orthanca: (1) listuje wszystkie studies, (2) dla wybranego wypisuje jego serie i
modalności, (3) pobiera jedną serię na dysk. Biblioteki do wyboru: czysty `requests`
albo `pydicom` + `requests`. Nie używaj gotowego SDK — chodzi o to, żebyś dotknął API.

> **Do przeczytania (Faza 1):**
> - Orthanc Book → „REST API” i „DICOMweb plugin”
> - orthanc-server/orthanc-setup-samples na GitHubie — **zajrzyj po naukę**, jak ludzie
>   konfigurują, ale nie kopiuj w całości

**✅ Checkpoint Fazy 1:** wrzuciłeś to samo badanie dwoma drogami (C-STORE i drag&drop),
widzisz je w web UI Orthanca, i Twój `explore.py` pobiera serię wyłącznie przez API.

---

## Faza 2 — Podłącz przeglądarkę OHIF

Cel: zobaczyć swoje badanie w profesjonalnym viewerze w przeglądarce. Tu nauczysz się,
jak frontend medyczny czyta dane (DICOMweb) i po co jest reverse proxy / CORS.

### 2.1 Dwie drogi — wybierz świadomie

- **Droga łatwa (na start):** obraz `orthancteam/orthanc` zawiera wbudowaną wtyczkę
  **OHIF**. Włączasz ją w configu i masz viewer pod `/ohif/` bez osobnego kontenera.
  Wtyczka ustawia OHIF na DICOMweb Orthanca automatycznie.
- **Droga „prawdziwa” (polecam po tym, jak łatwa zadziała):** osobny kontener OHIF,
  skonfigurowany ręcznie tak, by jako `dataSource` wskazywał DICOMweb Orthanca
  (QIDO/WADO/STOW pod `/dicom-web/`). Wtedy naprawdę rozumiesz, co się dzieje.

Konfiguracja OHIF sprowadza się do jednego obiektu `window.config` z listą
`dataSources` typu `dicomweb`, gdzie podajesz `qidoRoot` / `wadoRoot` wskazujące na
Orthanca. Napisz go sam na bazie docs OHIF — nie wklejaj cudzego `.js` w ciemno.

### 2.2 Pułapka, na której polegnie każdy: CORS

OHIF (jedna domena/port) pyta Orthanca (inna domena/port) → przeglądarka blokuje
z powodu CORS. Dwa wyjścia: (a) postaw **nginx jako reverse proxy**, który serwuje
OHIF i Orthanca spod jednego origin (to robi „prawdziwe” wdrożenie), albo (b) na dev
skonfiguruj proxy w OHIF. Zrozum, **dlaczego** to jest problem — to wróci w każdym
projekcie webowym.

**Zadanie 2.2.** Uruchom OHIF, wejdź w badanie, wykonaj pomiar (linijka/elipsa)
w viewerze. W dzienniku opisz, którą drogą (proxy/CORS) rozwiązałeś komunikację i dlaczego.

> **Do przeczytania (Faza 2):**
> - docs.ohif.org → „Data Sources → DICOMweb” i „Configuration”
> - Orthanc Book → „OHIF plugin”

**✅ Checkpoint Fazy 2:** Twoje badanie z Fazy 1 wyświetla się w OHIF w przeglądarce,
umiesz przewijać serię i zrobić pomiar.

---

## Faza 3 — Serce projektu: model, który je DICOM i wypluwa DICOM-SEG

To najtrudniejsza i najważniejsza faza. Tu wykorzystujesz model z inżynierki. Podejdź
etapami — nie próbuj od razu robić wszystkiego naraz.

### 3.1 Zrozum, dlaczego to nie jest „załaduj obraz i zrób forward”

Twój model trenował się na przygotowanych wolumenach (pewnie NIfTI, znormalizowane
spacing/orientacja). Ale wejściem z PACS jest **cała seria DICOM** — dziesiątki/setki
instancji, z metadanymi, których spacing może się różnić od tego, czego oczekuje model.
Pipeline produkcyjny musi:

```
seria DICOM  →  wybór właściwej serii (reguły)  →  złożenie w wolumen 3D
            →  preprocessing (spacing, orientacja, intensywność)  →  inferencja
            →  postprocessing  →  zapis jako DICOM-SEG odwołujący się do serii źródłowej
```

To dokładnie ten łańcuch, który w realnych ogłoszeniach nazywa się „budowa pipeline'ów
DICOM”.

### 3.2 Wybór narzędzia — dwie ścieżki, świadomie

**Ścieżka A — MONAI Deploy App SDK (polecam, bo uczy właściwych abstrakcji i pasuje
do CV).** SDK ma gotowe *operatory*, które składasz w aplikację (graf operatorów):

- `DICOMDataLoaderOperator` — wczytuje instancje DICOM
- `DICOMSeriesSelectorOperator` — wybiera właściwą serię wg reguł (np. Modality=MG)
- `DICOMSeriesToVolumeOperator` — składa serię w wolumen 3D
- `MonaiBundleInferenceOperator` — odpala Twój model **spakowany jako MONAI Bundle**
  (pre/postprocessing bierze z metadanych Bundle'a — spójrz, jak to elegancko domyka
  Projekt 1 z planu!)
- `DICOMSegmentationWriterOperator` — zapisuje wynik jako DICOM-SEG odwołujący się do
  serii źródłowej

Twoja praca: opakować model z inżynierki w **MONAI Bundle** (wagi TorchScript +
konfiguracja transformacji + metadane), a potem złożyć te operatory w aplikację. To
kilkadziesiąt linijek „kleju”, ale każda linijka czegoś uczy.

```python
# app.py — szkielet aplikacji MONAI Deploy (uzupełnij TODO)
from monai.deploy.core import Application
# TODO: zaimportuj operatory wymienione wyżej

class BreastSegApp(Application):
    def compose(self):
        loader   = ...  # DICOMDataLoaderOperator
        selector = ...  # DICOMSeriesSelectorOperator  (TODO: reguła wyboru serii)
        to_vol   = ...  # DICOMSeriesToVolumeOperator
        infer    = ...  # MonaiBundleInferenceOperator  (TODO: wskaż swój Bundle)
        seg_out  = ...  # DICOMSegmentationWriterOperator
                        # TODO: SegmentDescription — opis segmentu (nazwa, kod SNOMED)
        # TODO: połącz operatory krawędziami (self.add_flow(a, b, {...}))

if __name__ == "__main__":
    BreastSegApp().run()
```

**Ścieżka B — zrób to „ręcznie” raz, dla zrozumienia.** Zanim użyjesz SDK, napisz
minimalny wariant sam: `pydicom` do wczytania serii → złożenie wolumenu numpy →
Twój `torch` forward → maska. Do zapisu DICOM-SEG **nie pisz enkodera od zera** (to
pułapka na tygodnie) — użyj biblioteki **`highdicom`** (`highdicom.seg.Segmentation`),
która robi to poprawnie i zgodnie ze standardem. Ta ścieżka boli, ale po niej naprawdę
rozumiesz, co SDK robi za Ciebie.

> Rekomendacja: zrób **B w minimalnej wersji** (dla zrozumienia), a docelowy pipeline
> oprzyj na **A**. W README opisz, czego nauczyła Cię wersja ręczna — to pokazuje głębię.

### 3.3 Konwersja i dane testowe

Jeśli Twoje dane treningowe są w NIfTI, a potrzebujesz DICOM do testów pipeline'u:
`3D Slicer` konwertuje NIfTI→DICOM. Odwrotnie (`dcm2niix`) przyda się, gdy model chce
NIfTI. Zrozum, że **konwersja gubi część metadanych** — to realny problem kliniczny.

### 3.4 Weryfikacja wyniku — najważniejszy moment

Wygenerowany DICOM-SEG **wrzuć z powrotem do Orthanca** i otwórz w OHIF. Maska powinna
się nałożyć **dokładnie** na właściwe przekroje właściwej serii. Jeśli jest przesunięta,
odbita albo na złej serii — masz błąd w referencjach/orientacji. To najczęstsze źródło
frustracji i najlepsza lekcja o tym, jak precyzyjny jest DICOM.

**Zadanie 3.4.** Zmierz i zapisz: czas inferencji na jedną serię, zużycie pamięci GPU,
oraz Dice na jednym przypadku, dla którego masz ground truth. To pierwsze liczby do CV.
Na razie to tylko sanity check. Ewaluację, której da się wierzyć, zrobisz w Fazie 8.

> **Do przeczytania (Faza 3):**
> - MONAI Deploy App SDK docs → tutorial „Segmentation app” i „MONAI Bundle app”
> - `highdicom` docs → moduł `seg` (jeśli idziesz ścieżką B)
> - MONAI Bundle spec (domyka Projekt 1 z planu)

**✅ Checkpoint Fazy 3:** wrzucasz serię do Orthanca, ręcznie odpalasz swoją aplikację,
dostajesz DICOM-SEG, wrzucasz go do Orthanca i **widzisz poprawną nakładkę w OHIF**.
Masz pierwsze liczby (czas, pamięć, Dice).

---

## Faza 4 — Zamknij pętlę: automatyczne wyzwalanie

Do tej pory odpalasz model ręcznie. Teraz: badanie ląduje w PACS → segmentacja robi
się sama → wynik wraca do PACS. To zamienia „skrypt” w „system”.

### 4.1 Trzy sposoby (rosnąca dojrzałość)

1. **Orthanc Python/Lua plugin, hook `OnStoredInstance`/`OnStableStudy`.** Orthanc po
   ustabilizowaniu badania woła Twój kod. Najprostsze, wszystko w jednym miejscu.
2. **Osobny serwis-watcher.** Mały serwis (FastAPI) pyta Orthanca o nowe badania
   (albo dostaje webhooka), ściąga serię przez WADO, odpala inferencję, wysyła SEG przez
   STOW-RS z powrotem. Czystsza separacja — model nie żyje wewnątrz PACS.
3. **MONAI Deploy Informatics Gateway.** Dedykowany komponent, który przyjmuje DICOM
   (C-STORE/DICOMweb) i odpala aplikacje deploy. Najbliższe realnemu wdrożeniu.

Zrób **#1 albo #2** (start), a w README opisz #3 jako „jak by to wyglądało produkcyjnie”.
To pokazuje, że znasz drogę do dojrzałego rozwiązania.

### 4.2 Kolejka, nie blokowanie

Inferencja 3D trwa. Nie rób jej synchronicznie w hooku — **wrzuć zadanie do kolejki**
(Redis + prosty worker, albo Celery/RQ). Nauczysz się wzorca, który jest wszędzie
w produkcyjnym ML. Dodaj: retry przy błędzie, idempotencję (nie licz dwa razy tego
samego StudyInstanceUID).

**Zadanie 4.2.** Narysuj diagram sekwencji (choćby w Mermaid w README): modality →
Orthanc → kolejka → worker → SEG → Orthanc → OHIF. Rysowanie tego zmusza Cię do
zrozumienia przepływu.

**✅ Checkpoint Fazy 4:** robisz `storescu` jednego badania i **bez żadnej dalszej
akcji** po kilkunastu sekundach widzisz segmentację w OHIF. To jest moment „wow” projektu.

---

## Faza 5 — DICOM-SR: raport z liczbami

Segmentacja to obraz. Klinicysta chce też **wartości**. Dorzuć generowanie DICOM-SR
z: objętością zmiany (z maski i spacing), liczbą ognisk, ewentualnie pewnością modelu.

- Do zapisu SR znów użyj `highdicom` (`highdicom.sr`) — SR ma szablony (TID) i własną
  strukturę drzewa treści; pisanie od zera to droga przez mękę.
- Zacznij od minimalnego SR z jedną mierzoną wartością (objętość w ml). Potem rozbuduj.

Zrozum sens: **SEG mówi „gdzie”, SR mówi „ile”**. Razem to jest raport AI, który wraca
do systemu i jest maszynowo czytelny (można go potem agregować, audytować, porównywać).

**✅ Checkpoint Fazy 5:** dla przetworzonego badania w Orthancu leży zarówno DICOM-SEG,
jak i DICOM-SR z objętością zmiany; SR otwiera się w viewerze/Orthancu.

---

## Faza 6 — Warstwa produkcyjna (tu grasz swoją siłą)

To Twoja przewaga z CV (Docker, K8s, CI/CD, Grafana). Zrób ją porządnie — właśnie ta
warstwa odróżnia Cię od kandydatów „tylko-ML”.

### 6.1 CI/CD dla obrazu inferencyjnego
- GitHub Actions: build obrazu z modelem, testy, push do rejestru (GHCR).
- **Wersjonowanie modelu:** obraz taguj wersją modelu; zapisuj w SR/SEG, którą wersją
  algorytmu policzono wynik (`algorithm_version` — pole istnieje w DICOM-SEG nieprzypadkowo).
  To jest audytowalność, o którą pytają ogłoszenia.
- Na tym etapie model może być wpieczony w obraz. W Fazie 9 go z niego wyjmiesz: obraz
  będzie wersjonował kod, a rejestr modeli (MLflow) będzie wersjonował model.

### 6.2 Obserwowalność (Grafana — masz to)
Dashboard z metrykami, które naprawdę mają sens dla takiego systemu:
- czas inferencji na badanie (p50/p95),
- długość kolejki / czas oczekiwania,
- throughput (badania/godzinę),
- błędy inferencji, retry,
- (ambitnie) rozkład wielkości segmentu — wejście do monitoringu **driftu**.

Instrumentuj worker (Prometheus client) → Prometheus → Grafana. Znasz stack, więc to
pójdzie szybko; wartość jest w **doborze właściwych metryk**, nie w konfiguracji.

### 6.3 Monitoring driftu (wyróżnik)
Loguj statystyki wejścia (spacing, rozmiar, intensywności) i wyjścia (objętość
segmentu). Prosty alert, gdy rozkład odjeżdża od treningowego = sygnał driftu danych.
To temat, o którym większość juniorów nawet nie wie, że istnieje. Tu wystarczy zaczątek:
loguj cechy do pliku lub bazy. Porządny monitoring z testami statystycznymi i alertami
zbudujesz w Fazie 11.1 na tych samych logach.

### 6.4 Audit trail
Dla każdego przetworzonego badania zapisz wpis: StudyUID (zhashowany, nie PHI!), wersja
modelu, timestamp, czas inferencji, wynik. To Twoja „lineage” — słowo z ogłoszeń.

**✅ Checkpoint Fazy 6:** dashboard w Grafanie pokazuje latencję inferencji na badanie,
a każdy wynik ma zapisaną wersję modelu i wpis w audit logu.

---

# Część II — Cykl życia modelu: ML i MLOps

Po Części I masz system, który **uruchamia** model. Teraz zbudujesz wszystko, co dzieje
się przed inferencją i po niej: skąd model się bierze, skąd wiesz, że jest dobry, jak go
bezpiecznie wymienić na lepszy i jak zauważyć, że przestał działać. Część I pokazuje,
że rozumiesz workflow kliniczny. Część II pokazuje, że rozumiesz ML.

> **Zasada Części II:** każda liczba w README ma dać się odtworzyć jedną komendą,
> a każdy model w PACS-ie ma mieć numer wersji, który prowadzi do danych, kodu
> i metryk, z których powstał.

---

## Faza 7 — Dane jako kod: splity, wersjonowanie, powtarzalny trening

Cel: model przestaje być plikiem, który „kiedyś wytrenowałeś na inżynierce”. Każdy
trening da się powtórzyć: wiadomo, na jakich danych, z jaką konfiguracją i z którego
commitu powstał.

### 7.1 Manifest danych i split per pacjent

Najczęstszy błąd w medycznym ML to **leakage między pacjentami**. W mammografii każda
pierś ma co najmniej dwa rzuty (CC i MLO), pacjentka ma dwie piersi i często badania
z kilku lat. Jeśli dzielisz dane per obraz, ta sama zmiana trafia do treningu w rzucie
CC i do testu w rzucie MLO. Model „zna” ją z treningu, więc wynik testowy jest zawyżony.
W MRI jest tak samo: kilka serii z jednego badania, kilka badań jednej pacjentki.

Zbuduj `data/manifest.csv`, jedną linię na przypadek (obraz lub wolumen + maska):
`case_id`, `patient_id` (pseudonimizowany), `study_uid_hash`, ścieżka obrazu, ścieżka
maski, kolekcja źródłowa, producent aparatu (`Manufacturer`), gęstość piersi (jeśli jest),
rozmiar zmiany policzony z maski, `sha256` pliku. Z tego manifestu korzysta cała
Część II: splity, analiza podgrup, monitoring.

```python
# scripts/make_splits.py  (szkielet)
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def main(params):
    df = pd.read_csv("data/manifest.csv")
    # TODO: odfiltruj zbiór zewnętrzny (Faza 8.3) i golden set (Faza 9.2).
    #       One NIGDY nie trafiają do foldów.
    # TODO: wybierz kolumnę do stratyfikacji (np. producent albo przedział rozmiaru zmiany)
    sgkf = StratifiedGroupKFold(n_splits=params["n_folds"], shuffle=True,
                                random_state=params["seed"])
    # TODO: przypisz fold każdemu przypadkowi, grupując po patient_id
    # TODO: zapisz splits.json (case_id → fold) — to jest artefakt, nie efekt uboczny
```

**Zadanie 7.1.** Napisz test (pytest), który ładuje `splits.json` i manifest i sprawdza,
że żaden `patient_id` nie występuje w dwóch foldach ani w foldzie i zbiorze testowym
jednocześnie. Wepnij go w CI. Leakage ma łamać build, a nie wychodzić na rozmowie.
Jeśli masz wolny czas GPU: wytrenuj raz ze splitem per obraz i porównaj wynik walidacyjny.
Różnica to świetny akapit do README.

### 7.2 DVC: wersja danych w gicie, dane poza gitem

Dane nie idą do gita (są za duże i nie tam ich miejsce), ale ich **wersja** ma być
w gicie. DVC trzyma w repo mały plik `.dvc` z hashem, a same dane w zdalnym magazynie.
Masz już docker-compose, więc dołóż do niego **MinIO** jako magazyn zgodny z S3. Przyda
się też MLflow w 7.3.

```bash
dvc init
dvc remote add -d storage s3://dvc-store
dvc remote modify storage endpointurl http://localhost:9000   # TODO: MinIO z compose
dvc add data/raw                                              # → data/raw.dvc w gicie
```

Najważniejsze: **pipeline treningowy jako DAG w `dvc.yaml`**, a nie w notebooku.
`dvc repro` uruchamia tylko te etapy, których wejścia się zmieniły.

```yaml
# dvc.yaml  (szkielet)
stages:
  manifest:
    cmd: python scripts/build_manifest.py
    deps: [data/raw, scripts/build_manifest.py]
    outs: [data/manifest.csv]
  splits:
    cmd: python scripts/make_splits.py
    deps: [data/manifest.csv, scripts/make_splits.py]
    params: [split.seed, split.n_folds]
    outs: [data/splits.json]
  preprocess:
    # TODO: resampling do docelowego spacingu, orientacja, normalizacja
  train:
    # TODO: deps: dane po preprocessingu, splits.json, konfiguracja Bundla
    #       params: sekcja train z params.yaml;  outs: models/
  evaluate:
    # TODO: metrics: reports/metrics.json  (wypełnisz w Fazie 8)
```

Wszystkie hiperparametry trzymaj w `params.yaml`, a nie w argumentach wpisywanych
z ręki. Wtedy `dvc params diff` pokazuje, czym różnią się dwa eksperymenty.

> **Train/serve skew.** Preprocessing w treningu (`preprocess`) i w inferencji (Bundle
> z Fazy 3) musi pochodzić z **jednego źródła**. Jeśli trening robi resampling do 0,7 mm,
> a Bundle do 0,8 mm, model w PACS-ie działa gorzej niż na teście i nic tego nie
> sygnalizuje. Zdefiniuj transformacje raz, w konfiguracji Bundla, i używaj jej w obu
> miejscach.

### 7.3 MLflow: tracking eksperymentów

Postaw serwer MLflow w docker-compose. To jest Twoja strona DevOps: Postgres jako
backend store, MinIO jako magazyn artefaktów.

```yaml
# docker-compose.yml — dopisz  (szkielet)
  mlflow:
    image: ghcr.io/mlflow/mlflow:TODO-przypnij-wersje
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri postgresql://TODO
      --artifacts-destination s3://mlflow
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      # TODO: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY dla MinIO
    # Uwaga: oficjalny obraz może nie mieć sterownika Postgresa ani boto3.
    # Jeśli nie ma, zbuduj własny Dockerfile na jego bazie.
```

Każdy run ma zapisywać: wszystkie parametry, commit gita, wersję danych (hash z
`data/raw.dvc`), hash `splits.json`, metryki per epoka, metryki końcowe i artefakty
(konfiguracja Bundla, krzywe uczenia, model). Tagi: `fold`, `model_type` (`ours` /
`nnunet`), `trigger` (ręcznie / drift / nowe etykiety — przyda się w Fazie 11).

```python
# train.py — fragment  (szkielet)
import mlflow
from monai.utils import set_determinism

set_determinism(seed=params["train"]["seed"])
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast-seg")

with mlflow.start_run(run_name=f"unet-fold{fold}"):
    mlflow.log_params(...)              # TODO: spłaszczony params.yaml
    mlflow.set_tag("git_commit", ...)   # TODO: git rev-parse HEAD (i odmów treningu na brudnym drzewie)
    mlflow.set_tag("data_version", ...) # TODO: md5 z data/raw.dvc
    for epoch in range(...):
        ...
        mlflow.log_metric("val_dice", val_dice, step=epoch)
    # TODO: zaloguj Bundle jako artefakt (katalog) — w Fazie 9 zarejestrujesz go w rejestrze
```

Uczciwie o determinizmie: `set_determinism` nie gwarantuje identycznych wyników na GPU,
bo część operacji cuDNN jest niedeterministyczna. Zapisz w README, jaki rozrzut wyników
widzisz między powtórzeniami. To zresztą przydatna liczba: różnica między dwoma modelami
mniejsza niż ten rozrzut nic nie znaczy.

**Zadanie 7.3.** Usuń lokalnie `models/` i `data/`, zrób `git checkout <commit>`,
`dvc pull`, `dvc repro`. Sprawdź, czy odtworzony model ma metryki w granicach rozrzutu
od oryginału. Zapisz wynik w dzienniku.

> **Do przeczytania (Faza 7):**
> - DVC docs → „Get Started: Data Versioning” i „Data Pipelines”
> - MLflow docs → „Tracking” (serwer, backend store, artifact store)
> - Kapoor & Narayanan, „Leakage and the reproducibility crisis in machine-learning-based
>   science”, *Patterns* 2023 — katalog typów leakage, dobra lektura przed rozmową

**✅ Checkpoint Fazy 7:** `dvc repro` od zera odtwarza manifest, splity, trening
i ewaluację. Każdy run w MLflow ma commit, wersję danych i parametry. Test na brak
leakage między pacjentami przechodzi w CI.

---

## Faza 8 — Ewaluacja, której da się wierzyć

Cel: zamienić „Dice 0,82 na jednym przypadku” w tabelę, która obroni się przed
recenzentem. Na to rekruter ML patrzy najpierw.

### 8.1 Metryki właściwe dla segmentacji medycznej

Sam Dice ukrywa trzy rzeczy: błędy brzegu, pominięte **małe** zmiany (Dice liczony na
wolumenie jest zdominowany przez duże) i przypadki bez zmiany (gdy maska GT i predykcja
są puste, Dice jest nieokreślony). Dołóż:

- **HD95 w mm** (z uwzględnieniem spacingu) — błędy brzegu.
- **Metryki na poziomie zmiany:** czułość (zmiana GT jest wykryta, gdy pokrywa ją
  komponent predykcji powyżej progu IoU) i liczba fałszywych pozytywów na badanie.
  Radiologa obchodzi, czy zmiana została znaleziona, a nie Dice z dokładnością do 0,01.
- **Błąd objętości w ml.** Twój SR z Fazy 5 raportuje objętość, więc ewaluuj to, co
  raportujesz: bias i granice zgodności (wykres Blanda–Altmana).
- **Przypadki bez zmiany:** odsetek fałszywych alarmów.

```python
# eval/metrics.py  (szkielet)
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from scipy import ndimage

def case_metrics(pred, gt, spacing) -> dict:
    # TODO: Dice; ustal i ZAPISZ w README konwencję dla pustego GT i pustej predykcji
    # TODO: HD95 w mm — przekaż spacing, inaczej liczysz w wokselach
    # TODO: objętość pred i GT w ml (spacing!)
    ...

def lesion_metrics(pred, gt, iou_thr=0.1) -> dict:
    gt_cc, n_gt = ndimage.label(gt)
    pr_cc, n_pr = ndimage.label(pred)
    # TODO: TP = zmiany GT pokryte przez jakiś komponent predykcji (IoU > iou_thr)
    # TODO: FP = komponenty predykcji, które nie trafiają w żadną zmianę GT
    ...
```

Wynik zapisuj do `reports/per_case.csv`: jedna linia na przypadek plus kolumny
z manifestu (producent, gęstość, rozmiar zmiany). Z tego pliku liczysz wszystko dalej.
Ewaluacja to etap `evaluate` w `dvc.yaml`, więc `dvc metrics diff` porównuje dwa commity.

### 8.2 Przedziały ufności z bootstrapu

Na 50–100 przypadkach testowych różnica 0,82 vs 0,80 może być szumem. Każdą liczbę
w tabeli podawaj z 95% przedziałem ufności.

```python
# eval/stats.py  (szkielet)
import numpy as np

def bootstrap_ci(values, groups, n=2000, alpha=0.05, seed=0):
    # TODO: losuj ze zwracaniem PACJENTÓW (groups), nie przypadki — z tego samego
    #       powodu co w 7.1: przypadki jednej pacjentki są skorelowane
    # TODO: dla każdej próbki policz średnią; zwróć percentyle alpha/2 i 1-alpha/2
    ...

def paired_bootstrap_diff(values_a, values_b, groups, n=2000, seed=0):
    # TODO: te same wylosowane próbki dla obu modeli → rozkład różnicy średnich
    # TODO: jeśli przedział różnicy zawiera 0, NIE możesz napisać „model A jest lepszy”
    ...
```

`paired_bootstrap_diff` wróci w bramce jakości (Faza 9.2).

### 8.3 Zewnętrzny zbiór testowy: domain shift na własne oczy

Wybierz dane z **innego ośrodka lub producenta** niż treningowe i nie dotykaj ich aż do
końcowej ewaluacji: nie dobieraj na nich progów, nie wybieraj na nich checkpointu.
Przykładowe kierunki (sprawdź aktualną dostępność i licencje, bo się zmieniają):

- **Mammografia:** CBIS-DDSM (zdigitalizowane zdjęcia z kliszy, TCIA) i INbreast
  (mammografia cyfrowa, dostęp na wniosek). Klisza a obraz cyfrowy to bardzo wyraźny shift.
- **MRI piersi:** MAMA-MIA, zbiór złożony z kilku kolekcji TCIA z segmentacjami ekspertów.
  Pozwala zrobić „leave-one-source-out”: trenujesz na wszystkich źródłach poza jednym,
  a testujesz na tym jednym.

Raportuj wynik wewnętrzny i zewnętrzny obok siebie, oba z przedziałami ufności. Spadek
jest oczekiwany i to jest wartościowy wynik, a nie porażka. Opisz, skąd się bierze:
producent, protokół akwizycji, populacja.

**Zadanie 8.3.** Spróbuj jednej techniki łagodzenia shiftu: mocniejsza augmentacja
intensywności i kontrastu, inna normalizacja albo fine-tuning na małej części danych
zewnętrznych. W tym ostatnim przypadku wydziel z nich osobny, nietknięty test. Zmierz,
ile wyniku odzyskałeś.

### 8.4 Analiza podgrup i taksonomia błędów

Z `per_case.csv` zrób wykres Dice i czułości (z przedziałami ufności) per podgrupa:
producent, gęstość piersi (BI-RADS A–D, jeśli masz), rozmiar zmiany (kubełki), kolekcja.
Średnia z całego zbioru potrafi ukryć podgrupę, w której model jest bezużyteczny.

Potem otwórz w OHIF 10–15 najgorszych przypadków (to Twój pipeline z Części I, więc
wystarczy je wrzucić do Orthanca) i **skategoryzuj błędy**: pominięta mała zmiana,
fałszywy pozytyw na węźle chłonnym lub brodawce, zły brzeg, błąd w danych albo
w etykiecie. Tabela „typ błędu → liczba przypadków → hipoteza, co by pomogło” to materiał,
o który dopytuje się na rozmowie.

### 8.5 Baseline: nnU-Net

W segmentacji medycznej **nnU-Net** jest punktem odniesienia. Wiele „nowych architektur”
go nie pobija, jeśli porównanie jest uczciwe (Isensee i in., 2024). Wynik bez baseline'u
nic nie mówi. „Mój model vs nnU-Net na tych samych foldach” mówi dużo.

```bash
# nnU-Net v2  (szkielet — ścieżki i ID ustal sam)
export nnUNet_raw=TODO nnUNet_preprocessed=TODO nnUNet_results=TODO
# TODO: skonwertuj dane do formatu nnU-Net (DatasetXXX_Breast: imagesTr, labelsTr, dataset.json)
nnUNetv2_plan_and_preprocess -d XXX --verify_dataset_integrity
# TODO: NADPISZ splits_final.json w katalogu preprocessed foldami z Twojego splits.json
nnUNetv2_train XXX 3d_fullres 0      # dla mammografii konfiguracja 2d
```

> **Pułapka:** nnU-Net domyślnie robi własny losowy podział na 5 foldów **per przypadek**,
> a nie per pacjent. Bez nadpisania `splits_final.json` porównujesz modele na różnych
> foldach, a nnU-Net dodatkowo ma leakage. Pełny trening jest długi (domyślnie 1000 epok
> na fold). Na start wystarczy jeden fold albo krótszy wariant trenera, np.
> `-tr nnUNetTrainer_250epochs` (sprawdź listę wariantów w repo).

Predykcje nnU-Neta oceniaj **swoim** kodem z 8.1, a nie jego podsumowaniem, żeby
metryki były liczone identycznie. Loguj je do MLflow z tagiem `model_type=nnunet`.

Docelowa tabela w README:

| Model | Dice wewn. (95% CI) | HD95 [mm] | Czułość zmian | FP / badanie | Dice zewn. (95% CI) |
|---|---|---|---|---|---|
| Mój model | | | | | |
| nnU-Net | | | | | |

Jeśli nnU-Net wygrywa, napisz to wprost i pokaż, co daje Twój model (czas inferencji,
rozmiar, pamięć). Uczciwość w tej tabeli jest warta więcej niż lepsza liczba.

> **Do przeczytania (Faza 8):**
> - Maier-Hein i in., „Metrics reloaded: recommendations for image analysis validation”,
>   *Nature Methods* 2024 — jak dobrać metryki do problemu
> - Isensee i in., „nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image
>   Segmentation”, MICCAI 2024
> - README repozytorium MIC-DKFZ/nnUNet → format danych i ręczne splity

**✅ Checkpoint Fazy 8:** masz tabelę z przedziałami ufności dla swojego modelu i nnU-Neta,
na teście wewnętrznym i zewnętrznym, wykres podgrup i taksonomię błędów. Wszystko
odtwarza `dvc repro`.

---

## Faza 9 — Rejestr modeli, bramka jakości i bezpieczne wdrażanie

Cel: model przestaje być plikiem wpieczonym w obraz Dockera. Staje się wersjonowanym
obiektem z metrykami, który przechodzi automatyczną kontrolę, zanim dotknie PACS-a,
i którego wymianę można cofnąć jednym krokiem.

### 9.1 MLflow Model Registry

Każdy wytrenowany model (Bundle) rejestrujesz jako kolejną wersję modelu `breast-seg`.
Rolami steruje się **aliasami**:

- `champion` — wersja, której wynik trafia do PACS-a,
- `challenger` — kandydat, który przeszedł bramkę i działa w trybie shadow (9.4).

Worker przy starcie pobiera model po aliasie `champion`, ale w DICOM-SEG, SR i audit
logu zapisuje **numer wersji**, a nie alias. Alias się przesuwa, numer wersji nie.
Za pół roku z numeru wersji dojdziesz do runu w MLflow, a stamtąd do commitu i wersji
danych. To jest pełny lineage.

```python
# worker/model_loader.py  (szkielet)
import mlflow
from mlflow import MlflowClient

MODEL_NAME = "breast-seg"

def load_model(alias: str = "champion"):
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    # TODO: pobierz artefakty tej wersji (Bundle) do lokalnego cache
    # TODO: zwróć (ścieżka_do_bundla, mv.version) — wersja trafia do SEG/SR i audit logu
    # TODO: co jeśli rejestr jest niedostępny? (cache ostatniej wersji + głośny log)
```

Od tej chwili obraz Dockera wersjonuje **kod**, a rejestr wersjonuje **model**. To dwa
niezależne cykle życia: zmiana modelu nie wymaga przebudowy obrazu.

Do każdej wersji dołącz **model card** jako artefakt: przeznaczenie (i czego model nie
robi), dane treningowe z wersją DVC, metryki z Fazy 8 razem z podgrupami, znane
ograniczenia (np. „nie walidowany na tomosyntezie”). Tego samego rodzaju dokumentacji
oczekuje się od wyrobu medycznego.

### 9.2 Bramka jakości w CI

**Golden set:** 20–50 zamrożonych przypadków wersjonowanych w DVC, rozłożonych po
podgrupach, z kilkoma trudnymi (bez zmiany, bardzo mała zmiana, inny producent). Nie
trenujesz na nich i nigdy ich nie zmieniasz. Jeśli musisz je zmienić, tworzysz nową
wersję golden setu i przeliczasz championa.

Reguły trzymaj w `gate.yaml`, nie w kodzie:

- Dice i czułość zmian powyżej progów bezwzględnych,
- **non-inferiority względem championa:** dolna granica 95% CI różnicy Dice (paired
  bootstrap z 8.2) większa niż −margines, np. −0,02,
- czułość nie spada w żadnej podgrupie o więcej niż ustalony próg,
- latencja p95 w budżecie,
- test geometrii (9.3) przechodzi.

```python
# gate/check.py  (szkielet)
def main(candidate_version: str, champion_version: str) -> int:
    # TODO: wczytaj gate.yaml
    # TODO: policz per_case.csv obu wersji na golden secie (albo weź championa z cache)
    # TODO: paired bootstrap różnicy Dice → CI
    # TODO: sprawdź każdą regułę, zbierz listę naruszeń z czytelnym opisem
    # TODO: raport markdown → $GITHUB_STEP_SUMMARY
    # TODO: OK → ustaw alias "challenger" na kandydata i zwróć 0; nie OK → zwróć 1
    ...
```

Workflow w GitHub Actions uruchamiaj po rejestracji nowej wersji (`workflow_dispatch`
albo `repository_dispatch` z numerem wersji, wywołany przez pipeline treningowy).
Inferencja 3D w CI potrzebuje GPU, więc masz dwie opcje: self-hosted runner z GPU
albo mały golden set liczony na CPU (wolno, ale działa).

Bramka **nigdy nie promuje do `champion`**. Najwyżej przesuwa kandydata na `challenger`.
O zmianie modelu działającego na pacjentach decyduje człowiek (9.4).

### 9.3 Test regresji geometrii (end-to-end)

Ten test łapie dwie pułapki z listy na końcu przewodnika („zła orientacja/spacing” i „SEG
bez referencji do serii”), zanim zobaczy je radiolog.

W CI podnieś docker-compose (Orthanc i worker), wyślij znane badanie przez C-STORE,
poczekaj na SEG i porównaj go z referencyjnym:

- `ReferencedSeriesInstanceUID` wskazuje serię źródłową,
- liczba ramek i ich pozycje (per-frame `ImagePositionPatient` albo referencje do
  instancji źródłowych) się zgadzają,
- maska ma Dice ≥ 0,99 z referencyjną. Nie wymagaj 1,0, bo GPU nie jest w pełni
  deterministyczne. Ustal tolerancję świadomie i zapisz, dlaczego taka,
- `SegmentAlgorithmType` = `AUTOMATIC`, a wersja algorytmu = wersja z rejestru.

```python
# tests/e2e/test_geometry.py  (szkielet)
import highdicom as hd

def test_seg_matches_reference(orthanc, known_study):
    # TODO: wyślij known_study do Orthanca przez C-STORE (storescu albo pynetdicom)
    # TODO: odpytuj QIDO-RS, aż pojawi się seria Modality=SEG dla tego
    #       StudyInstanceUID (z timeoutem!)
    seg = hd.seg.segread(...)   # TODO: plik pobrany przez WADO-RS
    ref = hd.seg.segread("tests/data/reference_seg.dcm")
    # TODO: asercje z listy powyżej
```

**Test negatywny:** celowo odwróć jedną oś w preprocessingu i sprawdź, że test pada.
Test, który nigdy nie pada, niczego nie testuje.

### 9.4 Shadow deployment

Worker liczy wynik dwóch wersji: `champion` (wynik idzie do PACS-a) i `challenger`
(wynik idzie tylko do logów, MinIO i metryk, **nigdy do PACS-a**). Porównujesz na
żywych danych: zgodność masek (Dice champion vs challenger), różnicę objętości,
latencję i błędy. Dodaj w Grafanie panel per wersja.

- Challenger liczy się w **osobnej kolejce o niższym priorytecie**, żeby nie opóźniał
  wyniku klinicznego.
- Badania, na których wersje najbardziej się różnią, to naturalni kandydaci do przeglądu
  przez radiologa (Faza 11.3).
- Kryteria promocji zapisz z góry: minimum N badań w shadow, zero błędów, zgodność
  powyżej progu, a gdy masz etykiety z pętli zwrotnej (11.2), challenger nie jest gorszy.
- **Promocja** to przestawienie aliasu `champion`. Zrób ją jako workflow z ręczną
  akceptacją (GitHub Environments z wymaganym recenzentem). **Rollback** to przestawienie
  aliasu z powrotem. Przećwicz oba.

To zastępuje dawny stretch goal „A/B dwóch wersji”. W medycynie shadow jest
bezpieczniejszy niż A/B, bo pacjent nigdy nie dostaje wyniku z nieprzetestowanej wersji.

**Zadanie 9.4.** Zarejestruj celowo słaby model (np. po 2 epokach). Sprawdź, że bramka
go zatrzymuje z czytelnym powodem. Potem wepchnij go ręcznie na `challenger` i zobacz,
po ilu badaniach dashboard shadow pokazuje problem. Zapisz to w dzienniku.

> **Do przeczytania (Faza 9):**
> - MLflow docs → „Model Registry” (aliasy wersji)
> - GitHub docs → „Using environments for deployment” (required reviewers)
> - Sculley i in., „Hidden Technical Debt in Machine Learning Systems”, NeurIPS 2015
> - FDA, wytyczne o „Predetermined Change Control Plan” dla oprogramowania z AI — dlaczego
>   zmiana modelu w wyrobie medycznym wymaga zaplanowanej, udokumentowanej procedury

**✅ Checkpoint Fazy 9:** worker ładuje model z rejestru i zapisuje numer wersji w SEG/SR.
Słaby model zatrzymuje bramka w CI. Test geometrii przechodzi i pada na celowo zepsutej
orientacji. Challenger działa w shadow z panelem porównawczym. Promocję i rollback
robisz jednym krokiem.

---

## Faza 10 — Niepewność modelu i szybka inferencja

Cel: model mówi, kiedy nie należy mu ufać, i liczy szybciej bez utraty jakości,
co masz potwierdzone liczbami.

### 10.1 Niepewność

Model zawsze coś zwraca, także dla badania spoza rozkładu treningowego. Wynik bez miary
pewności nie mówi, kiedy trzeba go sprawdzić. Trzy metody, od najtańszej:

- **Test-time augmentation (TTA):** N przebiegów na odbitych lub lekko obróconych
  wersjach wejścia, potem rozrzut wyników. Nie wymaga retreningu.
- **MC dropout:** kilka przebiegów z włączonym dropoutem. Wymaga dropoutu w architekturze.
  Uwaga: włącz tryb treningowy **tylko** dla warstw dropout. `model.train()` na całym
  modelu przełącza też BatchNorm i psuje wynik.
- **Ensemble:** modele z 5 foldów (masz je z Fazy 7). Zwykle najlepsza jakość, ale
  5× koszt inferencji.

Z N predykcji liczysz mapę per woksel (entropia średniej predykcji) i **wynik na poziomie
badania**, np. średnią entropię w obrębie zmiany albo współczynnik zmienności objętości
między przebiegami.

```python
# uncertainty.py  (szkielet)
import torch

def tta_predict(model, volume, n=8):
    probs = []
    for flip_axes in ...:   # TODO: zestaw odbić po osiach przestrzennych
        # TODO: odbij → sliding_window_inference → odbij z powrotem → softmax
        probs.append(...)
    probs = torch.stack(probs)
    mean = probs.mean(0)
    # TODO: entropia średniej predykcji (mapa per woksel)
    # TODO: objętość zmiany w każdym przebiegu → współczynnik zmienności
    return mean, entropy_map, case_score
```

**Najważniejsze: sprawdź, czy ta niepewność cokolwiek mówi.** Na teście z Fazy 8 policz
korelację Spearmana między wynikiem niepewności a Dice i narysuj **krzywą risk-coverage**:
gdy odrzucisz X% najbardziej niepewnych przypadków, jak rośnie średni Dice pozostałych.
Jeśli nie rośnie, Twoja „niepewność” to szum. Wtedy napisz to uczciwie w README.
Dodatkowo sprawdź kalibrację prawdopodobieństw (reliability diagram, ECE)
i ewentualnie temperature scaling na walidacji.

Wpięcie w pipeline:
- wynik niepewności trafia do DICOM-SR (Faza 5) jako dodatkowy pomiar. Dla własnego
  pojęcia użyj prywatnego schematu kodowania (designator zaczynający się od `99`),
- próg ustalony **na walidacji, nie na teście** daje flagę „do weryfikacji” w SR,
- ambitnie: mapa niepewności jako DICOM Parametric Map (`highdicom.pm`) wyświetlana
  w OHIF obok nakładki.

### 10.2 Optymalizacja: ONNX → TensorRT

Pułapka na start: **sliding window zostaje w Pythonie**. Eksportujesz tylko sieć, która
przetwarza jeden patch o stałym rozmiarze (`roi_size` z konfiguracji Bundla).

```python
# export.py  (szkielet)
import torch
net = ...   # TODO: wczytaj wagi z Bundla; net.eval()
dummy = torch.randn(1, C, *ROI_SIZE, device="cuda")   # TODO: C i ROI_SIZE z Bundla
torch.onnx.export(net, dummy, "model.onnx", opset_version=17,
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
# TODO: porównaj wyjście onnxruntime z PyTorch (np.allclose z rozsądną tolerancją)
```

```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 \
        --minShapes=input:TODO --optShapes=input:TODO --maxShapes=input:TODO
```

Silnik TensorRT jest związany z konkretnym GPU i wersją TensorRT. W rejestrze trzymaj
ONNX, a silnik buduj przy wdrożeniu, na docelowej maszynie.

Tabela benchmarku na golden secie:

| Wariant | Latencja / badanie p50 / p95 | Pamięć GPU | Dice (golden) | Δ Dice vs FP32 |
|---|---|---|---|---|
| PyTorch FP32 | | | | — |
| PyTorch AMP | | | | |
| ONNX Runtime (CUDA) | | | | |
| TensorRT FP16 | | | | |

Zoptymalizowany model to **nowa wersja** w rejestrze, więc przechodzi bramkę z 9.2.
FP16 potrafi zgubić małe zmiany przy progu decyzyjnym, a ostatnia kolumna tabeli ma
to pokazać.

### 10.3 Triton Inference Server

Rozdziel odpowiedzialności: **worker** obsługuje DICOM, pre- i postprocessing oraz
SEG/SR, a **Triton** tylko sieć (dynamiczne batchowanie, wiele wersji, metryki GPU).
Każde z nich skalujesz osobno.

```
model_repository/
└── breast_seg/
    ├── config.pbtxt
    └── 1/
        └── model.plan        # albo model.onnx
```

```protobuf
# config.pbtxt  (szkielet)
name: "breast_seg"
platform: "tensorrt_plan"        # TODO: albo "onnxruntime_onnx"
max_batch_size: 8
input  [{ name: "input",  data_type: TYPE_FP32, dims: [ TODO ] }]   # C, D, H, W bez batcha
output [{ name: "logits", data_type: TYPE_FP32, dims: [ TODO ] }]
dynamic_batching { }             # TODO: max_queue_delay_microseconds — poczytaj o kompromisie
```

W workerze `sliding_window_inference` dostaje jako `predictor` funkcję, która wysyła
batch patchy do Tritona przez `tritonclient.grpc`. Triton wystawia metryki Prometheusa
na porcie 8002, więc dodaj je do Grafany (wykorzystanie GPU, czas w kolejce). Triton
może serwować dwie wersje jednego modelu naraz, a to gotowa infrastruktura dla
championa i challengera z 9.4. Napisz skrypt wdrożeniowy, który pobiera ONNX z rejestru,
buduje silnik i kładzie go w katalogu o numerze wersji zgodnym z MLflow.

> **Do przeczytania (Faza 10):**
> - Mehrtash i in., „Confidence Calibration and Predictive Uncertainty Estimation for
>   Deep Medical Image Segmentation”, *IEEE TMI* 2020
> - MONAI docs → `TestTimeAugmentation`
> - Triton docs → „Model Repository”, „Model Configuration”, „Dynamic Batcher”
> - TensorRT docs → `trtexec`

**✅ Checkpoint Fazy 10:** SR zawiera wynik niepewności, a flaga „do weryfikacji” działa
z progiem ustalonym na walidacji. Masz krzywą risk-coverage. Tabela benchmarku pokazuje
latencję i Dice przed optymalizacją i po niej. Worker woła Tritona, a zoptymalizowana
wersja przeszła bramkę.

---

## Faza 11 — Monitoring, pętla zwrotna i retrening

Cel: zamknąć cykl życia modelu. System wie, kiedy zmieniają się dane, zbiera poprawki
radiologa, mierzy rzeczywistą jakość i potrafi (z akceptacją człowieka) wypuścić
lepszy model.

### 11.1 Drift wejścia i wyjścia (rozwinięcie 6.3)

**Rozkład referencyjny:** cechy policzone na teście wewnętrznym, zapisane jako artefakt
wersji modelu w MLflow (`reference.parquet`). **Rozkład bieżący:** okno ostatnich N badań
z logów workera. Cechy per badanie:

- **metadane:** `Manufacturer`, `ManufacturerModelName`, spacing, `SliceThickness`,
  rozmiar obrazu, dla MRI natężenie pola i sekwencja,
- **intensywności:** percentyle 1/50/99 przed normalizacją i po niej,
- **wyjście modelu:** objętość, liczba ognisk, wynik niepewności (Faza 10), odsetek
  pustych predykcji,
- **ambitnie:** embeddingi z bottlenecku enkodera (średnia globalna + PCA).

Testy: dla cech kategorycznych chi² lub PSI, dla ciągłych KS lub odległość Wassersteina,
dla embeddingów MMD albo **klasyfikator domeny** (jeśli prosty klasyfikator odróżnia
dane referencyjne od bieżących z AUC wyraźnie powyżej 0,5, masz drift). Evidently robi
testy per cecha i raport HTML.

```python
# monitoring/drift_job.py  (szkielet) — uruchamiany np. raz na dobę
# Uwaga: API Evidently zmieniało się między wersjami — pisz według docs swojej wersji.
def run(window_days: int = 7):
    ref = ...   # TODO: reference.parquet z artefaktów wersji championa (MLflow)
    cur = ...   # TODO: cechy z ostatnich window_days z logów workera (bez PHI!)
    # TODO: raport driftu → HTML do MinIO, link na dashboardzie w Grafanie
    # TODO: odsetek cech z driftem, AUC klasyfikatora domeny → Prometheus (Pushgateway)
```

Alert w Grafanie odpalaj, gdy drift utrzymuje się przez **dwa kolejne okna**. Unikniesz
alarmów od jednego nietypowego dnia.

**Zadanie 11.1: zasymuluj drift.** Przepuść przez pipeline badania z innej kolekcji
lub producenta (np. zbiór zewnętrzny z 8.3; tu wolno, bo tylko monitorujesz, a nie
trenujesz). Czy alert odpala? Po ilu badaniach? Które cechy reagują pierwsze? Potem
odwrotnie: przepuść dane z rozkładu treningowego. Alert nie powinien odpalić, bo
fałszywe alarmy też są metryką.

> **Drift to nie to samo co spadek jakości.** Drift mówi „coś się zmieniło”, a nie
> „model się pogorszył”. O jakości wiesz tylko wtedy, gdy masz etykiety, dlatego jest 11.2.

### 11.2 Human-in-the-loop: poprawki radiologa jako etykiety

Przepływ:

1. Radiolog otwiera badanie z SEG modelu i poprawia maskę.
2. Poprawka zapisuje się jako **nowy** DICOM-SEG i nie nadpisuje wyniku modelu.
   Poprawkę rozpoznajesz po `SegmentAlgorithmType` = `MANUAL` lub `SEMIAUTOMATIC`
   (model zapisuje `AUTOMATIC`) i po innym `SeriesDescription`.
3. Kolektor (zadanie cykliczne) znajduje nowe poprawki przez QIDO-RS, dopasowuje je do
   SEG modelu po serii referencyjnej i liczy Dice oraz różnicę objętości.
4. Metryki trafiają do Prometheusa jako jakość „w produkcji” per wersja modelu, a pary
   (obraz, poprawiona maska) do nowej wersji danych w DVC.

Narzędzie do poprawek: nowsze wersje OHIF 3.x mają edycję segmentacji. Sprawdź, czy
Twoja wersja zapisuje SEG z powrotem przez STOW-RS. Jeśli nie, użyj **MONAI Label**
(serwer + wtyczka do OHIF lub 3D Slicera, obsługuje DICOMweb jako źródło danych).

```python
# feedback/collector.py  (szkielet)
def collect():
    # TODO: QIDO-RS: nowe serie Modality=SEG od ostatniego uruchomienia
    # TODO: rozróżnij wynik modelu (AUTOMATIC) od poprawki (MANUAL/SEMIAUTOMATIC)
    #       — SegmentSequence[i].SegmentAlgorithmType
    # TODO: dla poprawki znajdź SEG modelu z tą samą ReferencedSeriesInstanceUID
    # TODO: Dice, Δ objętości, numer wersji modelu (odczytany z SEG modelu) → metryki
    # TODO: zapisz parę do data/feedback/ + wpis w manifeście (source="feedback")
    # TODO: idempotencja — zapamiętaj przetworzone SOPInstanceUID (jak w 4.2)
```

**Nie masz radiologa? Zasymuluj go.** Dla przypadków ze zbioru zewnętrznego wgraj ground
truth jako SEG z `SegmentAlgorithmType` = `MANUAL`. Kolektor nie widzi różnicy, a Ty
testujesz cały przepływ.

W Grafanie zrób panel „Dice model vs poprawka, per wersja modelu, w czasie” i zawsze
pokazuj obok **liczbę przypadków**. Bez n ta liczba nic nie znaczy.

> **Automation bias.** Radiolog, który widzi maskę modelu, poprawia ją mniej, niż
> narysowałby od zera, więc Dice (model, poprawka) jest zawyżony. Wspomnij o tym w README.
> W realnym wdrożeniu część przypadków anotuje się bez podglądu wyniku modelu jako grupę
> kontrolną.

### 11.3 Active learning: co dać radiologowi do anotacji

Czas radiologa to najdroższy zasób w tym systemie. Kolejka do anotacji powinna mieć
kolejność opartą na **niepewności** (Faza 10), **niezgodności champion–challenger**
(9.4) i **wyniku driftu** badania. Dołóż też kilka przypadków losowych, żeby estymata
jakości produkcyjnej nie była obciążona.

**Eksperyment offline** (na danych, dla których masz etykiety): zacznij od 20% zbioru
treningowego i w każdej rundzie dodawaj K przypadków wybranych (a) losowo albo
(b) według niepewności, a potem trenuj od nowa. Narysuj krzywą Dice vs liczba etykiet
dla obu strategii. Jeśli (b) nie bije (a), to też jest wynik. Sprawdź dlaczego, bo
często niepewność wskazuje przypadki z błędną etykietą albo artefaktami. To wiele
treningów, więc użyj krótkiego harmonogramu albo niższej rozdzielczości. Każdy run
loguj do MLflow.

### 11.4 Retrening z człowiekiem w pętli

Wyzwalacze: (a) N nowych etykiet z pętli zwrotnej, (b) alert driftu utrzymujący się
dłużej niż ustalony czas, (c) decyzja ręczna. Pipeline (flow w Prefect albo workflow
w GitHub Actions wołający `dvc repro`):

```
nowe etykiety / drift / ręcznie
   → nowa wersja danych (DVC)
   → dvc repro: splity → trening → ewaluacja (MLflow)
   → rejestracja wersji w MLflow
   → bramka w CI (9.2) → alias challenger
   → shadow (9.4)
   → decyzja człowieka → alias champion
```

```python
# pipelines/retrain.py  (szkielet)
from prefect import flow, task

@task
def snapshot_data(): ...        # TODO: dołącz data/feedback do manifestu; dvc add; commit + tag
@task
def train_and_eval(): ...       # TODO: dvc repro; zwróć run_id z MLflow
@task
def register(run_id): ...       # TODO: mlflow.register_model → numer wersji
@task
def trigger_gate(version): ...  # TODO: repository_dispatch do GitHub Actions (Faza 9.2)

@flow
def retrain(reason: str):
    # TODO: zapisz reason (drift / etykiety / ręcznie) jako tag runu — to też jest lineage
    ...
```

Dwie zasady dla danych z pętli zwrotnej:
- **Splity dalej per pacjent.** Nowy przypadek pacjentki, która jest już w teście,
  idzie do testu, a nie do treningu. Golden set i zbiór zewnętrzny zostają zamrożone.
- **Nie mierz jakości produkcyjnej na danych, na których trenujesz.** Gdy poprawki trafią
  do treningu, jakość nowej wersji mierzysz tylko na poprawkach, które przyszły później.

Dlaczego promocja nie jest automatyczna: w wyrobie medycznym zmiana modelu jest zmianą
wyrobu, a regulacje (MDR, wytyczne FDA o PCCP) wymagają zaplanowanych, zwalidowanych
zmian. W pełni automatyczna promocja do championa byłaby dla rekrutera z medtechu
czerwoną flagą. Napisz to w README jednym zdaniem.

> **Do przeczytania (Faza 11):**
> - Rabanser i in., „Failing Loudly: An Empirical Study of Methods for Detecting
>   Dataset Shift”, NeurIPS 2019
> - Evidently docs → raporty i presety driftu
> - MONAI Label docs → integracja z OHIF i DICOMweb
> - Google Cloud, „MLOps: Continuous delivery and automation pipelines in machine
>   learning” — poziomy dojrzałości MLOps. Porównaj, na którym poziomie jest Twój projekt

**✅ Checkpoint Fazy 11:** symulowany drift odpala alert, a dane z rozkładu treningowego
nie. Kolektor zbiera poprawki i Grafana pokazuje Dice model vs poprawka per wersja.
Masz za sobą jeden pełny obieg: nowe etykiety → retrening → bramka → shadow → ręczna
promocja → nowy champion w PACS-ie, a każdy krok zostawił ślad w gicie, DVC i MLflow.

---

# Część III — Portfolio

## Faza 12 — Refleksja i README (to robi z projektu portfolio)

Kod bez opisu nie istnieje dla rekrutera. Poświęć na to realny czas.

### 12.1 Porównaj się z checklistą MONAI Deploy
Przejdź dokumentację MONAI Deploy i wypisz w README: **co Twój pipeline ma, a czego
świadomie nie zrobiłeś i dlaczego**. To pokazuje, że znasz różnicę między projektem
edukacyjnym a wdrożeniem klinicznym — a to dojrzałość, której nie widać u juniorów.
Zrób to samo dla MLOps: porównaj projekt z poziomami dojrzałości z lektury do Fazy 11
i napisz, na którym jesteś i czego brakuje do następnego.

### 12.2 Napisz sekcję „Co by wymagało wdrożenie kliniczne”
Krótko, ale konkretnie. Walidacja zewnętrzna: masz jej zaczątek z Fazy 8.3, ale
wdrożenie wymaga prospektywnej walidacji w docelowym ośrodku. Zgodność z **IEC 62304**
(cykl życia oprogramowania medycznego) i **ISO 13485**, obsługa PHI i anonimizacja,
ślad regulacyjny pod **EU AI Act / MDR**, plan kontrolowanych zmian modelu (PCCP) jako
formalna wersja tego, co zbudowałeś w Fazach 9 i 11. Nie musisz tego wdrażać — musisz
**wiedzieć, że to istnieje**. To dwa akapity, które ustawiają Cię jako kogoś, kto myśli
o produkcie, nie o notebooku.

### 12.3 Struktura README (proponowana)
1. Jedno zdanie: co to robi (z GIF-em: badanie in → segmentacja w OHIF out).
2. Architektura: diagram Mermaid z Fazy 4 plus diagram cyklu życia modelu (Część II).
3. Jak uruchomić (`docker compose up`, gdzie wrzucić dane, gdzie kliknąć).
4. Co się dzieje pod spodem: DICOM → wolumen → inferencja → DICOM-SEG/SR → PACS.
5. **Wyniki:** tabela z Fazy 8 (przedziały ufności, zbiór zewnętrzny, nnU-Net), wykres
   podgrup, taksonomia błędów, benchmark z Fazy 10.
6. **Odtwarzalność:** `dvc pull && dvc repro`, link do eksperymentów w MLflow, model card.
7. **Cykl życia modelu:** rejestr, bramka, shadow, monitoring, pętla zwrotna, w tym co
   automatyczne, a co wymaga decyzji człowieka i dlaczego.
8. Decyzje projektowe i **czego się nauczyłem** (wyciąg z `JOURNAL.md`).
9. „Droga do produkcji” (12.1 + 12.2).
10. Ograniczenia i dane (jasno: dane publiczne, nie szpitalne; automation bias; rozmiar
    zbioru testowego).

### 12.4 Nagraj dwa krótkie dema
- **Demo kliniczne (60 s):** `storescu` → chwila → nakładka pojawia się w OHIF → SR
  z objętością i flagą niepewności.
- **Demo MLOps (60–90 s):** rejestracja słabego modelu → bramka w CI go zatrzymuje →
  dobry model przechodzi → panel shadow → promocja → rollback.

To rekruter obejrzy w 90 sekund i o to zapyta na rozmowie.

**✅ Checkpoint Fazy 12:** repo ma README z oboma diagramami, GIF-ami, tabelą wyników
z przedziałami ufności, sekcją „czego się nauczyłem” i „droga do produkcji”. Ktoś obcy
potrafi uruchomić pipeline z README i odtworzyć wyniki przez `dvc repro`.

---

## Mapa całości (jedno spojrzenie)

**Runtime (Części I + II):**

```
[modality / storescu]
        │ C-STORE (DIMSE)
        ▼
   ┌─────────┐  OnStableStudy  ┌─────────┐  job  ┌──────────┐  gRPC  ┌──────────────┐
   │ Orthanc │ ──────────────► │ kolejka │ ────► │  worker  │ ─────► │    Triton    │
   │ (PACS)  │                 └─────────┘       │ (DICOM,  │        │  champion +  │
   │         │ ◄── STOW-RS: SEG + SR ─────────── │ pre/post)│        │  challenger  │
   └────┬────┘     (tylko champion)              └────┬─────┘        └──────────────┘
        │ DICOMweb (QIDO/WADO/STOW)                   │ metryki, cechy driftu,
        ▼                                             │ wyniki shadow
   ┌─────────┐                                        ▼
   │  OHIF   │  ← nakładka + raport        [Prometheus / Grafana / Evidently]
   └────┬────┘
        │ poprawka radiologa (SEG, MANUAL) → Orthanc → [kolektor] → data/feedback (DVC)
```

**Cykl życia modelu (Część II):**

```
  dane + splity (DVC)  ◄─────────────────────────────────┐
        │                                                │
        ▼                                                │
  trening + ewaluacja (dvc repro → MLflow)               │
        │                                                │
        ▼                                                │  retrening
  MLflow Registry: wersja N                              │  (nowe etykiety,
        │                                                │   alert driftu
        ▼                                                │   albo ręcznie)
  bramka CI: golden set + test geometrii ──(nie)──► stop │
        │ (tak) → alias: challenger                      │
        ▼                                                │
  shadow: challenger liczy obok championa                │
        │                                                │
        ▼                                                │
  decyzja człowieka → alias: champion → PACS             │
        │                                                │
        ▼                                                │
  monitoring driftu + poprawki radiologa ────────────────┘
```

---

## Pułapki, które oszczędzą Ci wieczorów

**Pipeline kliniczny (Część I):**

- **DICOM-SEG bez referencji do serii źródłowej** → viewer nie pokaże nakładki albo
  pokaże ją nie na tym. Zawsze przekazuj metadane serii źródłowej do writer'a.
- **Zła orientacja/spacing** → maska przesunięta lub odbita. Sprawdzaj na znanym
  przypadku, nie „na oko” na losowym (a potem zautomatyzuj to w teście z 9.3).
- **CORS** (Faza 2) → nie walcz z przeglądarką, postaw reverse proxy.
- **Inferencja synchronicznie w hooku Orthanca** → blokuje PACS. Zawsze kolejka.
- **Pisanie enkodera DICOM-SEG/SR od zera** → tygodnie bólu. Używaj `highdicom` /
  operatorów MONAI Deploy.
- **Dane szpitalne w publicznym repo** → nigdy. Publiczna wersja tylko na zbiorach
  otwartych (TCIA). To też sygnał, że rozumiesz PHI.
- **Twarde kodowanie ścieżek/UID-ów** → wszystko przez konfigurację i metadane.

**ML i MLOps (Część II):**

- **Split per obraz zamiast per pacjent** → leakage i zawyżony wynik. Test w CI (7.1).
- **nnU-Net z domyślnymi splitami** → porównanie na innych foldach niż Twój model,
  a nnU-Net dodatkowo ma leakage. Nadpisz `splits_final.json`.
- **Dobór progu lub checkpointu na teście** (zwłaszcza zewnętrznym) → optymistyczny
  wynik, którego nie powtórzysz.
- **Inny preprocessing w treningu i w inferencji** (train/serve skew) → model w PACS-ie
  działa gorzej niż na teście i nic tego nie sygnalizuje.
- **Dice bez konwencji dla pustych przypadków** → średnia zależy od przypadku, a nie
  od modelu.
- **Alias zamiast numeru wersji w audit logu** → nie wiesz, który model policzył wynik.
- **Silnik TensorRT jako przenośny artefakt** → nie zadziała na innym GPU. Trzymaj ONNX.
- **`model.train()` dla MC dropout** → BatchNorm w trybie treningowym psuje predykcje.
- **Pomiar jakości produkcyjnej na danych, które trafiły do treningu** → wynik zawyżony.
- **Automatyczna promocja do championa** → w medycynie zawsze decyduje człowiek.

---

## Stretch goals (jak chcesz błysnąć)

- **Interaktywność:** zamiast tylko full-auto, pozwól klinicyście kliknąć prompt
  i dociąć maskę — wpięcie MedSAM2 jako drugiego trybu (łączy się z Projektem 3 z planu).
  Poprawki z tego trybu zasilają pętlę zwrotną z 11.2.
- **Wdrożenie na Kubernetes** zamiast compose (masz K8s w CV — pokaż to): worker jako
  deployment ze skalowaniem po długości kolejki (KEDA), Triton jako osobny deployment
  z GPU.
- **Helm chart** całości — jedno polecenie stawia PACS + viewer + kolejkę + model
  + MLflow + MinIO + Triton + monitoring.
- **Uczenie federacyjne:** potraktuj kolekcje źródłowe jako osobne „szpitale”
  i wytrenuj model bez łączenia danych (NVIDIA FLARE albo Flower). Porównaj z treningiem
  centralnym i z leave-one-source-out z 8.3.

---

## Co powiedzieć na rozmowie (jednym zdaniem)

> „Zbudowałem end-to-end pipeline, w którym badanie DICOM trafia do PACS-a (Orthanc),
> automatycznie uruchamia segmentację mojego modelu, a wynik wraca jako DICOM-SEG i
> DICOM-SR i wyświetla się w OHIF. Wokół tego zbudowałem cykl życia modelu: wersjonowane
> dane i splity per pacjent, ewaluację z przedziałami ufności na zewnętrznym zbiorze
> i z baseline'em nnU-Net, rejestr modeli z bramką jakości w CI i wdrożeniem shadow,
> monitoring driftu oraz pętlę zwrotną, w której poprawki radiologa stają się nowymi
> etykietami.”

Jeśli rozmowa jest stricte o MLOps, zacznij od drugiej połowy tego zdania, a PACS
potraktuj jako kontekst. Jeśli o imagingu, odwrotnie.

To zdanie, poparte działającym repo i dwoma krótkimi demami, robi z Ciebie kandydata,
który „rozumie workflow kliniczny i cykl życia modelu”, a nie „trenował U-Net”. Dokładnie
ten profil, na który jest niedobór ludzi.

---

*Powodzenia. Rób fazy po kolei, prowadź dziennik, commituj małe kroki. Kiedy utkniesz —
najpierw docs z sekcji „Do przeczytania”, potem eksperyment, dopiero potem pytanie.
To jest dokładnie ten sposób pracy, którego oczekuje się od inżyniera.*
