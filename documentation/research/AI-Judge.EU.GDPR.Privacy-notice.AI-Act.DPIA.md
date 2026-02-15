NL chatGPT notes

Ja — maar hier zitten **twee verschillende verwerkingen** met een andere AVG-impact. Ik zet ze even helder naast elkaar voor jouw use-case (jury-sport, actieherkenning, transparantie/opt-out).

---

## 1️⃣ Actieherkenning (skills classificeren, niet personen)

**Doel:** herkennen welke beweging/skill wordt uitgevoerd.

**Privacy-impact**

* Als het model **geen personen identificeert of volgt**, maar enkel patronen van beweging analyseert → meestal **geen of beperkte verwerking van persoonsgegevens**.
* Toch kan videomateriaal **indirect identificeerbaar** zijn (gezicht, lichaam, context). Dan val je wél onder de AVG.

**Rechtsgrond (meestal)**

* ✔ Gerechtvaardigd belang (bijv. sportanalyse, jurering verbeteren)
  → vereist belangenafweging + minimale gegevensverwerking.
* ✔ Overeenkomst (als deelnemers deelnemen aan een event met analyse als onderdeel).
* ❌ Toestemming alleen nodig als je verder gaat dan redelijk verwacht of gevoelige data gebruikt.

**Best practices**

* Anonimiseren of pseudonimiseren vóór training (blur gezichten, skeleton tracking).
* Geen persistent ID per persoon.
* Data-minimalisatie (alleen wat nodig is voor skillherkenning).
* Duidelijke uitleg in privacyverklaring.

👉 Dit scenario is privacy-technisch het minst risicovol.

---

## 2️⃣ Persoonsherkenning / similariteit / aanwezigheid tellen

**Doel:** detecteren of dezelfde persoon in meerdere video’s voorkomt (bv. voor transparantie of opt-out).

Hier verander je fundamenteel van categorie.

**Privacy-impact**

* Dit is **biometrische of identificeerbare persoonsverwerking**.
* Zelfs “similarity matching” zonder naam kan identificatie mogelijk maken → AVG volledig van toepassing.
* Als je unieke kenmerken van lichaam/gezicht gebruikt om iemand te onderscheiden → vaak **biometrische gegevens**.

**Rechtsgrond (realistisch)**

* ✔ Toestemming (sterk aanbevolen / vaak vereist bij biometrische identificatie).
* Soms gerechtvaardigd belang mogelijk, maar moeilijk te verdedigen bij herkenning van personen op video.

**Extra verplichtingen**

* DPIA (Data Protection Impact Assessment) is zeer waarschijnlijk nodig.
* Duidelijke opt-out én effectieve uitvoering.
* Beperkte bewaartermijnen.
* Sterke beveiliging.
* Transparantie over:

  * hoe matching werkt
  * hoe lang embeddings worden bewaard
  * of automatische besluitvorming plaatsvindt

👉 Dit scenario heeft **hoog privacyrisico** en vereist stevig juridisch kader.

---

## Belangrijke nuance voor jouw “transparantie-doel”

Je intentie (mensen laten zien in hoeveel video’s ze voorkomen en opt-out vergemakkelijken) is privacy-vriendelijk, maar:

* Het middel (automatische persoonsgroepering) is zélf een zware verwerking.
* Privacyrecht kijkt naar **wat je technisch doet**, niet alleen naar je bedoeling.

**Privacy-vriendelijkere alternatieven**

* Handmatige tagging op verzoek van betrokkene.
* Zelfidentificatie (persoon claimt eigen beelden).
* Tijdelijke matching zonder opslag van biometrische templates.
* Client-side matching (op toestel van gebruiker).

---

## Samengevat

✔ Actieclassificatie zonder persoonsidentificatie → meestal mogelijk zonder toestemming, mits zorgvuldig ingericht.
⚠ Persoonsherkenning/similarity → hoog risico, vaak toestemming + DPIA nodig.

---

Als je wil, kan ik helpen bepalen:

* of jouw huidige model technisch onder “biometrische verwerking” valt
* of een gerechtvaardigd belang verdedigbaar is voor jouw jurysport-context
* welke tekst in je privacyverklaring dit correct dekt

Vertel me hoe de video’s worden verzameld (publiek event? deelnemers? online uploads?) — dat bepaalt de juiste juridische route.

---

Helder — bij **video’s van een publiek/federatief georganiseerd sportevent** wordt de juridische basis meestal zo opgebouwd:

---

## 🟢 Actieherkenning voor jurering (zonder persoonsidentificatie)

Dit past doorgaans goed binnen de AVG als je het netjes inricht.

**Waarschijnlijke rechtsgrond**

* **Gerechtvaardigd belang** van de organisator/federatie (objectievere jurering, sportanalyse).
* Eventueel **uitvoering van de overeenkomst** met deelnemers (analyse als onderdeel van deelname).

**Wat je moet doen**

* Inschrijfformulier + privacyverklaring: expliciet vermelden dat video’s worden gebruikt voor AI-analyse van skills.
* **Data-minimalisatie**: gezichten blurren of skeleton tracking gebruiken vóór training.
* Geen blijvende identifiers per persoon.
* Redelijke bewaartermijnen en goede beveiliging.
* Mogelijkheid tot bezwaar waar dat praktisch kan.

👉 In deze vorm is **geen toestemming nodig**, zolang je niet identificeert of profileert op persoonsniveau.

---

## 🟠 Persoonsherkenning / similariteit om aanwezigheid te tellen

Omdat het event publiek is, betekent dat **niet** dat je vrij bent om mensen automatisch te herkennen of te groeperen.

**Waarom dit zwaarder is**

* Je gaat personen **onderscheiden of volgen** over video’s heen.
* Dat kan biometrische verwerking worden (gezicht/lichamelijke kenmerken).
* Hoge privacy-impact → vaak **toestemming** nodig of anders een zeer sterke onderbouwing.

**Als je dit toch wilt**

* Voer een **DPIA** uit vóór je start.
* Overweeg **expliciete toestemming** van deelnemers (bij inschrijving, apart van algemene voorwaarden).
* Sla geen biometrische templates op langer dan strikt nodig.
* Maak de **opt-out echt effectief** (verwijderen uit datasets, retraining-beleid, duidelijke procedure).
* Leg transparant uit: doel, techniek, bewaartermijn, rechten.

👉 Voor toeschouwers in beeld is toestemming praktisch lastig → liever **niet automatisch herkennen**.

---

## 🟡 Praktische, privacy-vriendelijke alternatieven voor jouw transparantie-doel

Als je mensen wilt helpen zien waar ze in voorkomen en makkelijk laten verwijderen:

* **Self-claim portal**: persoon uploadt een foto of selecteert fragmenten om eigen beelden te vinden (matching tijdelijk, zonder opslag).
* **Handmatige of semi-automatische tagging** op verzoek.
* **Client-side matching** (zoekproces op het apparaat van de gebruiker).
* Publiceer een **heldere verwijderingsworkflow** met snelle responstermijn.

---

## Kort samengevat

* Skill-/actieherkenning voor jurering bij een publiek sportevent → meestal oké zonder toestemming, mits niet-identificerend en goed uitgelegd.
* Automatische persoonsherkenning of groepering → hoog risico; reken op DPIA en (vaak) expliciete toestemming, zeker voor deelnemers; voor publiek liever vermijden.

Als je wil, schets ik een voorbeeldparagraaf voor je privacyverklaring die precies past bij deze event-context.

---
Ja — dat model **kan** AVG-conform, maar alleen als je het technisch en organisatorisch strak afbakent.

## ✔ Wat juridisch werkt

**Persoonsherkenning uitsluitend voor deelnemers die expliciet toestemming geven**, en alle anderen standaard als “Onbekend” behandelen, past in principe binnen de AVG omdat:

* De **rechtsgrond = toestemming** (vrij, specifiek, geïnformeerd, ondubbelzinnig).
* Je **niet probeert** niet-toestemmers te identificeren.
* Je doel beperkt is (bv. transparantie over waar iemand voorkomt, opt-out faciliteren).

## ⚠ Cruciale randvoorwaarden

Om dit echt verdedigbaar te maken, heb je deze waarborgen nodig:

### 1) Strikte scheiding in de pipeline

* Herkenningsmodel draait **alleen** tegen de whitelist van personen met toestemming.
* Geen opslag van embeddings/templates van niet-toestemmers.
* Default output voor alle anderen = **Onbekend** zonder verdere profilering.

### 2) Toestemming die standhoudt

* Apart van algemene voorwaarden (geen “bundeling”).
* Duidelijk doel: “herkenning voor transparantie/opt-out”.
* **Intrekken moet net zo makkelijk** zijn als geven, met effect op datasets en toekomstige trainingen.

### 3) Data-minimalisatie & bewaartermijnen

* Bewaar biometrische templates zo kort mogelijk.
* Overweeg **on-device of tijdelijke matching** i.p.v. centrale opslag.
* Retentiebeleid voor video’s én modelartefacten (incl. retraining-procedure na verwijdering).

### 4) DPIA en transparantie

* Voer een **DPIA** uit (hoog risico door biometrische kenmerken).
* Privacyverklaring + event-communicatie: hoe herkenning werkt, wie meedoet, hoe je bezwaar maakt.

### 5) Bescherming van omstanders

* Voor toeschouwers: geen identificatie, geen templates, en waar redelijk **blur/pseudonimiseer**.

## 👍 Praktische implementatie (privacy-vriendelijk)

* **Whitelist-matching**: alleen vergelijken met embeddings van toestemmers.
* **Confidence-drempel**: bij twijfel → Onbekend.
* **Self-service portaal**: deelnemer ziet eigen hits en kan verwijderen.
* **Auditlogs**: wie/wanneer herkenning is uitgevoerd (zonder extra persoonsgegevens te creëren).

## Samengevat

Ja, herkenning voor alleen toestemmende personen met alle anderen als “Onbekend” is een werkbare route — mits je toestemming robuust is, je niemand anders identificeert, en je DPIA + minimalisatie serieus doorvoert.

Wil je dat ik een korte toestemmingsclausule en DPIA-checklist op maat voor jouw jurysport-event schets?

---

Goede vragen — je zit precies op de grens waar techniek en AVG elkaar raken. Ik pak ze één voor één, praktisch en juridisch.

---

## 🔹 1) Blurren zonder persoonsidentificatie — hoe doe je dat “AVG-proof”?

De kern: **je mag mensen detecteren om ze te anonimiseren**, zolang die detectie niet wordt gebruikt om ze te herkennen of te volgen.

**Privacy-veilige methodes (aanbevolen volgorde)**

### ✅ Skeleton / pose extraction (beste optie)

* Detecteer enkel gewrichten/poses (stick-figure representatie).
* Verwijder originele pixels vóór opslag of training.
* Voordeel: skills blijven analyseerbaar, identiteit verdwijnt.
* Veel gebruikt in sportanalyse.

### ✅ Full-body / face blur met directe discard

* Gebruik een detector (persoon/gezicht) → blur → **origineel frame meteen weg**.
* Geen tracking-ID’s over frames heen.
* Geen opslag van crops of templates.

### ✅ Silhouet of segmentation masking

* Persoon wordt een uniform vlak (of transparant) i.p.v. blur.
* Nog minder risico op heridentificatie via kleding/omgeving.

**Wat je beter vermijdt**

* Persistent tracking-ID per persoon.
* Opslag van niet-geblurde video’s voor “later”.
* Face recognition libraries (ook niet “per ongeluk”).

**Documenteer dit**

* In je DPIA: “detectie enkel voor anonimisering; geen identificatie; geen templates”.

---

## 🔹 2) Valt skill-analyse ook onder AI-jurering?

Ja — maar juridisch is er een verschil tussen **assistentie** en **autonome besluitvorming**.

### ✔ Jury-assistentie (laag tot middel risico)

AI geeft:

* scorevoorstellen,
* segmentatie van acties,
* vergelijkingen met referenties,
* consistentiechecks.

**Menselijke jury beslist eindscore.**
Dit is doorgaans te onderbouwen met gerechtvaardigd belang / overeenkomst, mits transparantie.

### ⚠ (Semi-)autonome jurering

AI bepaalt de score, jury valideert vooral.

Dan kom je in de buurt van:

* **geautomatiseerde besluitvorming** met significante gevolgen.
* Extra verplichtingen: uitlegbaarheid, betwistingsmogelijkheid, menselijke tussenkomst.

**Advies voor jouw fase**
Omschrijf het als:

> “AI-ondersteunde jurering, waarbij de uiteindelijke beslissing door menselijke juryleden wordt genomen.”

Dat houdt je juridisch in de veilige zone terwijl je technisch kunt opschalen.

---

## 🔹 3) Embeddings opslaan vs. model weights

Goed dat je dit scherp hebt — het is een groot verschil.

### Embeddings

* Vectorrepresentaties van een persoon of fragment.
* Bij persoonsherkenning vaak **biometrisch**.
* Opslag = hogere privacy-impact.

👉 Jij slaat ze niet op → dat is privacyvriendelijk.

### Model weights

* Geleerde parameters van het model.
* Bevatten normaal **geen direct identificeerbare personen**.
* Maar: bij kleine datasets kan theoretisch memorisatie optreden.

**Best practice**

* Train op geanonimiseerde input (pose/blur).
* Vermijd overfitting (regularisatie, data-augmentatie).
* Beperk toegang tot trainingsdata.

---

## 🔹 4) Praktische architectuur voor jouw use-case

Een privacy-sterke pipeline zou er zo uitzien:

1️⃣ Inname video (eventcontext)
2️⃣ Detectie persoon → pose/silhouet/blur
3️⃣ Origineel beeld discard of streng beperkt bewaren
4️⃣ Skill-classificatie op geanonimiseerde data
5️⃣ AI-score = advies
6️⃣ Menselijke jury beslist
7️⃣ Alleen model weights bewaren, geen embeddings

Voor opt-out:

* Verwijder bronvideo + retrain of exclude bij volgende training.

---

## 🔹 5) Wat je nu al goed doet

* Geen opslag van embeddings.
* Focus op skill i.p.v. persoon.
* Transparantie/opt-out als doel.

Dat plaatst je al in een **relatief lage privacy-impact** categorie, zolang je identificatie vermijdt.

---

Als je wil, kan ik:

* een korte DPIA-structuur maken specifiek voor jouw pipeline, of
* voorbeeldtekst geven voor “AI-ondersteunde jurering” in je privacyverklaring.
