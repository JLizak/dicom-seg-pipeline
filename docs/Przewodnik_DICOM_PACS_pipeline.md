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

**Realny budżet czasu:** 3–4 weekendy, jeśli robisz solo i uczciwie czytasz docs.
Faza 0–2 to jeden weekend. Faza 3 to serce projektu i zajmie najwięcej. Faza 6 gra
na Twoją siłę (DevOps), więc pójdzie szybko.

---

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
To temat, o którym większość juniorów nawet nie wie, że istnieje.

### 6.4 Audit trail
Dla każdego przetworzonego badania zapisz wpis: StudyUID (zhashowany, nie PHI!), wersja
modelu, timestamp, czas inferencji, wynik. To Twoja „lineage” — słowo z ogłoszeń.

**✅ Checkpoint Fazy 6:** dashboard w Grafanie pokazuje latencję inferencji na badanie,
a każdy wynik ma zapisaną wersję modelu i wpis w audit logu.

---

## Faza 7 — Refleksja i README (to robi z projektu portfolio)

Kod bez opisu nie istnieje dla rekrutera. Poświęć na to realny czas.

### 7.1 Porównaj się z checklistą MONAI Deploy
Przejdź dokumentację MONAI Deploy i wypisz w README: **co Twój pipeline ma, a czego
świadomie nie zrobiłeś i dlaczego**. To pokazuje, że znasz różnicę między projektem
edukacyjnym a wdrożeniem klinicznym — a to dojrzałość, której nie widać u juniorów.

### 7.2 Napisz sekcję „Co by wymagało wdrożenie kliniczne”
Krótko, ale konkretnie: walidacja na zewnętrznym zbiorze (domain shift!), zgodność
z **IEC 62304** (cykl życia oprogramowania medycznego) i **ISO 13485**, obsługa PHI /
anonimizacja, ślad regulacyjny pod **EU AI Act / MDR**. Nie musisz tego wdrażać —
musisz **wiedzieć, że to istnieje**. To dwa akapity, które ustawiają Cię jako kogoś,
kto myśli o produkcie, nie o notebooku.

### 7.3 Struktura README (proponowana)
1. Jedno zdanie: co to robi (z GIF-em: badanie in → segmentacja w OHIF out).
2. Architektura (diagram Mermaid z Fazy 4).
3. Jak uruchomić (`docker compose up`, gdzie wrzucić dane, gdzie kliknąć).
4. Co się dzieje pod spodem: DICOM → wolumen → inferencja → DICOM-SEG/SR → PACS.
5. Metryki: tabela (czas inferencji, pamięć, Dice na przykładzie).
6. Decyzje projektowe i **czego się nauczyłem** (wyciąg z `JOURNAL.md`).
7. „Droga do produkcji” (7.1 + 7.2).
8. Ograniczenia i dane (jasno: dane publiczne, nie szpitalne).

### 7.4 Nagraj 60-sekundowe demo
GIF/wideo: `storescu` → chwila → nakładka pojawia się w OHIF → SR z objętością. To jest
to, co rekruter obejrzy w 90 sekund i o czym zapyta na rozmowie.

**✅ Checkpoint Fazy 7:** repo ma README z diagramem, GIF-em, tabelą metryk, sekcją
„czego się nauczyłem” i „droga do produkcji”. Ktoś obcy potrafi to uruchomić z README.

---

## Mapa całości (jedno spojrzenie)

```
[modality / storescu]
        │ C-STORE (DIMSE)
        ▼
   ┌─────────┐   OnStableStudy    ┌──────────┐   job    ┌────────┐
   │ Orthanc │ ─────────────────► │  kolejka │ ───────► │ worker │
   │ (PACS)  │                    └──────────┘          │ (model)│
   │         │ ◄─ STOW-RS: SEG+SR ───────────────────────┘
   └────┬────┘
        │ DICOMweb (QIDO/WADO)
        ▼
   ┌─────────┐
   │  OHIF   │  ← tu widzisz nakładkę + raport
   └─────────┘
        ▲
   [Prometheus/Grafana] ← metryki z workera (latencja, kolejka, drift)
```

---

## Pułapki, które oszczędzą Ci wieczorów

- **DICOM-SEG bez referencji do serii źródłowej** → viewer nie pokaże nakładki albo
  pokaże ją nie na tym. Zawsze przekazuj metadane serii źródłowej do writer'a.
- **Zła orientacja/spacing** → maska przesunięta lub odbita. Sprawdzaj na znanym
  przypadku, nie „na oko” na losowym.
- **CORS** (Faza 2) → nie walcz z przeglądarką, postaw reverse proxy.
- **Inferencja synchronicznie w hooku Orthanca** → blokuje PACS. Zawsze kolejka.
- **Pisanie enkodera DICOM-SEG/SR od zera** → tygodnie bólu. Używaj `highdicom` /
  operatorów MONAI Deploy.
- **Dane szpitalne w publicznym repo** → nigdy. Publiczna wersja tylko na zbiorach
  otwartych (TCIA). To też sygnał, że rozumiesz PHI.
- **Twarde kodowanie ścieżek/UID-ów** → wszystko przez konfigurację i metadane.

---

## Stretch goals (jak chcesz błysnąć)

- **Interaktywność:** zamiast tylko full-auto, pozwól klinicyście kliknąć prompt
  i dociąć maskę — wpięcie MedSAM2 jako drugiego trybu (łączy się z Projektem 3 z planu).
- **Wdrożenie na Kubernetes** zamiast compose (masz K8s w CV — pokaż to): worker jako
  deployment ze skalowaniem po długości kolejki (KEDA).
- **Helm chart** całości — jedno polecenie stawia PACS + viewer + kolejkę + model.
- **A/B dwóch wersji modelu** z porównaniem metryk na dashboardzie.

---

## Co powiedzieć na rozmowie (jednym zdaniem)

> „Zbudowałem end-to-end pipeline, w którym badanie DICOM trafia do PACS-a (Orthanc),
> automatycznie uruchamia segmentację mojego modelu, a wynik wraca jako DICOM-SEG i
> DICOM-SR i wyświetla się w OHIF — z wersjonowaniem modelu, kolejką, dashboardem
> latencji w Grafanie i zaczątkiem monitoringu driftu.”

To zdanie, poparte działającym repo i 60-sekundowym demem, robi z Ciebie kandydata,
który „rozumie workflow kliniczny”, a nie „trenował U-Net”. Dokładnie ten profil, na
który jest niedobór ludzi.

---

*Powodzenia. Rób fazy po kolei, prowadź dziennik, commituj małe kroki. Kiedy utkniesz —
najpierw docs z sekcji „Do przeczytania”, potem eksperyment, dopiero potem pytanie.
To jest dokładnie ten sposób pracy, którego oczekuje się od inżyniera.*
