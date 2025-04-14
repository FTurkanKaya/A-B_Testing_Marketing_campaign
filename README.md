# A/B Test Analyse van Advertentiecampagnes
Deze repository bevat een analyse van een A/B-test die de effectiviteit van advertentiecampagnes vergelijkt. Het evalueert de conversieratio's tussen de "ad"- en "PSA"-groepen met behulp van statistische methoden zoals normaliteitstests, variantieanalyse en de Mann-Whitney U-test om de impact van advertenties op conversiesuccessen te beoordelen.

## Dataset
De dataset bestaat uit twee groepen:

## Ad Groep: Blootgesteld aan advertenties.
## PSA Groep: Blootgesteld aan publieke serviceberichten (geen advertenties).

De belangrijkste variabele voor de analyse is de conversieratio, die het percentage gebruikers vertegenwoordigt dat de gewenste actie (conversie) ondernam na blootstelling aan de behandeling (advertenties of PSA).

## Belangrijke Stappen in de Analyse
## Normaliteitstest:

Shapiro-Wilk en Kolmogorov-Smirnov tests werden uitgevoerd om te controleren of de data een normale verdeling volgt. Beide tests gaven aan dat de data niet normaal verdeeld is.
## Test van Variantiegelijkheid:
De Levene-test werd uitgevoerd om te controleren of de variances van de twee groepen gelijk zijn. Het resultaat gaf aan dat de variances ongelijk zijn, wat het gebruik van niet-parametrische tests suggereert.

## Mann-Whitney U Test:
Deze test werd gebruikt om de conversieratio’s van de twee groepen te vergelijken. Er werd een significant verschil gevonden, wat aangeeft dat de ad-groep een hogere conversieratio had dan de PSA-groep.

## Resultaten
Ad Groep: 2,55% conversieratio.
PSA Groep: 1,79% conversieratio.

De Mann-Whitney U-test toonde een statistisch significant verschil tussen de twee groepen, waarbij de ad-groep beter presteerde in termen van conversieratio.

## Conclusie & Strategische Aanbevelingen
Campagnesucces: De advertentiecampagne was succesvol, aangezien deze resulteerde in een significant hogere conversieratio dan de PSA-groep.

## Advertentie-impact: Het verschil in conversieratio’s kan worden toegeschreven aan de aanwezigheid van advertenties, aangezien de beide groepen verder vergelijkbaar waren.

## Optimalisatie-aanbevelingen:

Verhoog de investering in advertenties, aangezien deze leiden tot een hogere conversieratio.
Optimaliseer de timing van advertenties voor de beste prestaties (bijvoorbeeld in de middag en doordeweeks).
Verdere analyses kunnen het testen van advertentiefrequentie en gebruikerssegmentatie omvatten.

## Gebruikte Hulpmiddelen & Bibliotheken
pandas: Data manipulatie
numpy: Numerieke bewerkingen
scipy: Statistische tests (Shapiro-Wilk, KS-test, Levene’s test, Mann-Whitney U test)
matplotlib / seaborn: Data visualisatie
