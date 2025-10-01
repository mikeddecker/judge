# Jurysporten en AI skill herkenning; gedaan met subjectiviteit?

Skillherkenning door AI staat voor de deur. In jurysporten zoals rope skipping, artistieke gymnastiek of synchroonzwemmen zou er binnenkort wel eens een machinale gestuurde jury-assitent de routine mee kunnen beoordelen en zo de score van de oefening bepalen.

De Decker Mike, een student toegepaste informatica aan de Hogeschool in Gent (Hogent), creëerde voor zijn bachelorproef een jury assistent die rope skipping skills herkende in double dutch 3 (DD3) freestyles. DD3 is een soort evenement in rope skipping, waarbij er twee draaiers zijn en één springer.

(foto DD3 example)

## Wat is rope skipping?

Rope skipping is net als artistieke gymnastiek of dans een sport met veel creativiteit en verschillende elementen. Om te bepalen wie de beste, mooiste en moeilijkste routine of freestyle heeft, is er jury panel die volgens een juryhandleiding de freestyle live of vertraagd herbekijken om een numerieke score toe te kennen. Aangezien dit soms voor subjectiviteit of menselijke fouten kan zorgen, kan een computer gestuurde assistent hierbij helpen, idealiter onder het toeziend oog van een menselijke jury expert.

## Hoe werkt de assistent?

Naar voorbeelden van beeldherkenning, gebarentaalherkenning of NextJump's speedcounter onderzocht Mike in zijn bachelorproef de mogelijheid tot het herkennen van meerdere verschillende skills in een volledige video. Hiervoor waren een drie grote obstakels, die meteen de rode draad in de gehele architectuur van het AI model vormden.

Rope skipping freestyles zijn routines van 60 tot 75 seconden die een aaneenschakeling zijn van verschillende skills, die niet even lang duren of niet altijd ingezoomd op de springers...

## Stap 1 van de 3: Lokaliseren van de skippers

De eerste stap is het lokaliseren van de athleten. Deze zijn vaak gefilmd met statische camera's waardoor ze de ene keer dicht of verder weg van de camera springen. Hierdoor kunnen ze uitgeknipt worden en kan elke pixel optimaal benut worden in de volgende stappen. Denk maar aan je gsm die naast je computer dezelfde video afspeelt. Het is toch makkelijker te zien wat er zich afspeelt op je computer?

(Afbeelding localize - before & after)

## Stap 2: Segmenteren van korte skill fragmenten.

Eenmaal ingezoomd kunnen we bepalen wanneer een skill start en eindigt. Dit concept valt in rope skipping tamelijk mee. Doorgaans begint en eindigt een skill wanneer de springer de grond verlaat en weer op de grond land. Immers moet er een touw onder kunnen. Deze momenten aanduiden en laten leren door de computer, zorgt voor een fragmenteerbare freestyles.

(Afbeelding fragmentjes/grafiek)

## Stap 3: Herkennen van de skill

Eenmaal we verschillende fragmenten hebben, kunnen we het AI model laten leren, bereken of raden welke skill er getoond wordt. Twee opmerkingen.

Ten eerste bestaan skills uit verschillende elementen. Zo heb je 3 atleten, waarvan 2 draaiers en 1 springer en geef je aan wat de springer doet, wat draaier 1 doet, draaier 2, het aantal touwrotaties etc. Het model gokt/berekent dan respectievelijk alle verschillende elementen, op basis van wat het reeds zag/leerde.

Ten tweede variëren skills in lengte. De tijdspanne die een handstand is verschillend van een salto, noch doe je zelf nog eens een handstand, dan is de tweede keer langer of korter dan de eerste. In essentie is een video een opeenvolging van afbeeldingen. Stel dat de handstand 18 afbeeldingen zijn en de salto 15, dan kan je alle fragmenten gelijk maken naar 16 afbeeldingen door enkele 2 van de 18 afbeeldingen te verwijderen en ééntje van de 15 te dupliceren. Geef deze aan het model die bepaalt welke skill er getoont wordt.

(Afbeelding/timelaps skill)

## Post AI skill herkenning

Eenmaal je weet welke skills in de freestyle zitten, kan je ze omzetten naar een numerieke score volgens de vooropgestelde juryhandleiding. Onder toeziend oog van een jury expert, kunnen zo freestyles transparanter, objectiever, correcter en door minder juryleden gejureerd worden.

Deze technologie maakt het tevens mogelijk om een live score te geven tijdens de freestyle of om het aantrekkelijker te maken voor het publiek, door vertraagde fragmentjes opnieuw af te spelen met skillbeschrijving erbij.

> Hoewel er nog veel fout voorspellingen zijn, door een gelimiteerde gelabelde dataset, begint de AI al basiselementen zoals een plankhouding (push-up), handstand (frog), split of salto te herkennen.

(Video, predicted skills)

## Toekomst van de assistent

Als je mijn bachelorproef zou lezen, lees je dat er nog veel optimalisatiemogelijkheden zijn. Zo is er mogelijk idee om de lokalisatiestap met de segmentatiestap te combineren en zo veel meer.

Een interessante optimalisatie momenteel in testfase is het idee om het model toepasbaar te maken op meerdere events en sport overschrijdend in één gehele dataset. Hiervoor zijn er nog onvoldoende nieuwe labels, zelfs geen van andere sporten om hiervoor resultaat te tonen. De grafiek hieronder toont alleszins de stijgende accuraatheid van de herkenningsfase in de set-up van de bachelorproef die enkel focuste op één event, namelijk double dutch single freestyle (DD3).

(Grafiek accuracy=buiten context hier)

Deze technologie is niet enkel interessant voor rope skipping, maar ook voor gymnastiek, synchroonzwemmen, kunstschaatsen etc.

