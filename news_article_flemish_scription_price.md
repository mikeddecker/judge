# Jurysporten en AI skillherkenning: gedaan met subjectiviteit?

Skillherkenning door Artificiële Intelligentie staat voor de deur. In jurysporten zoals rope skipping, artistieke gymnastiek of synchroonzwemmen zou er binnenkort wel eens een machinaal gestuurde jury-assistent de routine kunnen beoordelen en zo helpen bij het bepalen van de score van de oefening.

Mike De Decker, een student toegepaste informatica aan de Hogeschool in Gent (Hogent), creëerde voor zijn bachelorproef een juryassistent die rope skipping skills herkende in Double Dutch 3 (DD3) freestyles. DD3 is een specifiek onderdeel in rope skipping, waarbij er twee draaiers zijn en één springer. Je kan het vergelijken met de balk of ongelijke leggers in de gymnastiek of het hordelopen, verspringen of discuswerpen in de atletiek.

(foto DD3 example)

## Wat is rope skipping?

Rope skipping is net als artistieke gymnastiek of dans een sport met veel creativiteit en verschillende elementen. Om te bepalen wie de beste, mooiste en moeilijkste routine of freestyle heeft, is er een jurypanel die volgens een juryhandleiding de freestyle live of vertraagd bekijken om een numerieke score toe te kennen. Aangezien dit soms voor subjectiviteit of menselijke fouten kan zorgen, kan een computergestuurde assistent hierbij helpen, idealiter onder het toeziend oog van een menselijke jury-expert.

## Hoe werkt de assistent?

Naar voorbeelden van beeldherkenning, gebarentaalherkenning of NextJump's speedcounter onderzocht Mike in zijn bachelorproef de mogelijkheid tot het herkennen van meerdere verschillende skills in een volledige video. Hierbij waren er drie grote obstakels, die de rode draad in de gehele architectuur van het AI-model vormden. Rope skipping freestyles duren namelijk 60 tot 75 seconden en zijn een aaneenschakeling van verschillende skills, die variabel zijn in lengte en op verschillende afstand worden gefilmd. Hieronder worden de drie grote stappen kort toegelicht. 

## Stap 1: Lokaliseren van de skippers

De eerste stap is het lokaliseren van de atleten. Deze zijn vaak gefilmd met statische camera's, of op verschillende zoomlevels, waardoor ze de ene keer dicht of verder weg van de camera springen. Hierdoor kunnen ze uitgeknipt worden en kan elke pixel optimaal benut worden in de volgende stappen. Denk maar aan tegelijkertijd een video afspelen op je gsm en een computer. Het is toch makkelijker te zien wat er zich afspeelt op je computer?

(Afbeelding localize - before & after)

## Stap 2: Segmenteren van korte skill fragmenten.

Eenmaal ingezoomd kunnen we bepalen wanneer een skill start en eindigt. Dit concept is relatief eenvoudig, want doorgaans begint en eindigt een skill wanneer de springer de grond verlaat en weer op de grond landt. Immers moet er een touw onder kunnen. Deze momenten aanduiden en laten leren door de computer, zorgt voor fragmenteerbare freestyles.

(Afbeelding fragmentjes/grafiek)

## Stap 3: Herkennen van de skill

Eenmaal we verschillende fragmenten hebben, kunnen we het AI model voorbeeld skills laten leren en ongeziene skills laten voorspellen. Twee opmerkingen.

Ten eerste bestaan skills uit verschillende elementen. Zo heb je 3 atleten, waarvan 2 draaiers en 1 springer en geef je aan wat de springer doet, wat draaier één doet, wat draaier twee, hoeveel touwrotaties er waren... Het model berekent dan alle verschillende elementen, op basis van wat het leerde uit voorbeeldskills.

De tweede opmerking gaat over de skills die variëren in lengte. De tijdspanne die een handstand inneemt is verschillend van een salto. Daarbij duurt niet iedere handstand even lang. In essentie is een video een opeenvolging van afbeeldingen. Stel dat het handstand fragment 18 afbeeldingen zijn en de salto 15, dan kan je alle fragmenten gelijk maken naar 16 afbeeldingen, door 2 van de 18 afbeeldingen te verwijderen en ééntje van de 15 te dupliceren. Geef deze 16 afbeeldingen aan het model om te berekenen welke skill er getoond werd.

(Afbeelding/timelaps skill)

## Post AI skill herkenning

Eenmaal je weet welke skills in de freestyle zitten, kan je ze omzetten naar een numerieke score volgens de vooropgestelde juryhandleiding. Onder toeziend oog van een jury expert, kunnen freestyles transparanter, objectiever, accurater en door minder juryleden beoordeeld worden.

Deze technologie maakt het tevens mogelijk om een live score te geven tijdens de freestyle of om het aantrekkelijker te maken voor het publiek, door skillfragmenten opnieuw af te spelen met naam erbij.

> Hoewel er nog veel foutieve voorspellingen zijn, door een gelimiteerde gelabelde dataset, begint de AI al basiselementen zoals een plankhouding (push-up), handstand (frog), split of salto te herkennen.

(Video, predicted skills)

## Toekomst van de assistent

Als je mijn bachelorproef zou lezen, merk je dat er nog veel optimalisatiemogelijkheden zijn. Zo is er mogelijk idee om de lokalisatiestap met de segmentatiestap te combineren en zo veel meer.

Een interessante optimalisatie momenteel in testfase is het idee om het model toepasbaar te maken op meerdere events en sportoverschrijdend in één gehele dataset. Echter zijn er nog onvoldoende labels, laat staan labels van andere sporten. De grafiek hieronder toont de stijgende accuraatheid van de herkenningsfase in de set-up van de bachelorproef die enkel focuste op één event, namelijk Double Dutch single freestyle (DD3). Daarbij heeft niet elke skill voldoende voorbeelden om uit te leren, aldus is hogere accuraatheid uitgesloten. 

(Afbeelding: Grafiek van de accuracy)

Deze technologie is niet enkel interessant voor rope skipping, maar ook voor gymnastiek, synchroonzwemmen, kunstschaatsen en andere sporten. Verder is het een flexibele technologie die aangepast kan worden naar de vereiste toepassing.

