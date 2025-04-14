#####################################################################
#####################################################################
##                          Marketing A/B TESTING                  ##
#####################################################################
#####################################################################

"""
Zakelijke Probleem

Marketingbedrijven willen begrijpen of hun reclamecampagnes daadwerkelijk effectief zijn.
Hiervoor gebruiken ze vaak A/B-testen. In deze dataset is een deel van de gebruikers blootgesteld aan advertenties,
terwijl een ander deel enkel publieke dienstmededelingen (PSA) te zien kreeg. Het centrale zakelijke probleem omvat de volgende vragen:

--->>  Was de reclamecampagne succesvol?

--->>  In hoeverre is het succes toe te schrijven aan de advertenties?

De antwoorden op deze vragen helpen bedrijven om hun marketingbudgetten efficiënter in te zetten.

#***********************************************************************
Datasetbeschrijving:

De dataset bevat informatie over 588.101 gebruikers die deelnamen aan een digitale marketingcampagne.
Voor elke gebruiker zijn de volgende gegevens beschikbaar:

user id: Unieke identificatie van de gebruiker.
test group: De groep waartoe de gebruiker behoort. “ad” betekent dat de gebruiker advertenties heeft gezien;
            “psa” betekent dat hij alleen een publieke mededeling zag.

converted: Of de gebruiker het product heeft gekocht (True/False).
total ads: Het totale aantal advertenties dat de gebruiker heeft gezien.

most ads day: De dag waarop de gebruiker de meeste advertenties heeft gezien.

most ads hour: Het uur van de dag waarop de meeste advertenties werden bekeken (0–23).

#**************************************************************************
Metric Definities:


Conversieratio (Conversion Rate): 	Het percentage gebruikers binnen een groep dat het product heeft gekocht.
                                    Formule: aantal converted = True / totaal aantal gebruikers

Lift                            :	Het verschil in conversieratio tussen de advertentiegroep en de controlegroep.
                                    Formule: CR(ad) - CR(psa)

Relatieve Lift                  :	Het procentuele verschil in conversieratio tussen advertentiegroep en controlegroep.
                                    Formule: (CR(ad) - CR(psa)) / CR(psa) × 100

Exposure (Blootstelling)        :	Het aantal advertenties dat een gebruiker heeft gezien (total ads).
Most Ads Day                    :	De dag waarop de gebruiker de meeste advertenties zag.
Most Ads Hour                   :	Het uur van de dag waarop de meeste advertenties zijn bekeken (0–23).
"""

##########################################
# Gegevensvoorbereiding en Analyse
##########################################
# Stap 1:
# Lees de gegevens van de controle- en testgroep uit het bestand 'marketing_AB.csv'.
# Wijs de gegevens van de controle- en testgroep toe aan afzonderlijke variabelen.
#*******************************************************

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import kstest


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print(os.getcwd())
file = os.getcwd()
filePath = "Git_Project/dataset/marketing_AB.csv"

df = pd.read_csv(filePath)
df.head()

# 'ad' ve 'psa' gruplarını filtreleyelim
ad_group = df[df['test group'] == 'ad']
psa_group = df[df['test group'] == 'psa']


##########################################
# Stap 2:
# Bereken de conversieratio voor elke groep
# In een A/A-test ontvangen beide groepen dezelfde behandeling
# (d.w.z. er zijn geen verschillende inhouden zoals "ad" en "psa" of de inhoud die getest wordt wordt op dezelfde manier weergegeven).
# Het doel van deze test is:

# Testen of het systeem (bijvoorbeeld gebruikerssegmentatie, willekeurige distributie of analysemethoden) goed werkt.
# Als de conversieratio's van beide groepen erg dicht bij elkaar liggen en er geen statistisch significante verschillen zijn,
# geeft dit aan dat het systeem goed werkt (d.w.z. de groepen zijn willekeurig verdeeld onder gelijke omstandigheden).

#*******************************************************
# Conversieratio van de groepen
#******************************************************
ad_conversion_rate = ad_group['converted'].mean()
psa_conversion_rate = psa_group['converted'].mean()

print(f"Ad Group Conversion Rate: {ad_conversion_rate}")
print(f"PSA Group Conversion Rate: {psa_conversion_rate}")

#***********************************************************************************************
#  Onderzoek de dataset verder:
#  bijvoorbeeld, is de "total ads" verdeling gelijk,
#  zijn er onevenwichtigheden in metrics zoals "most ads day"?

# 1- Verdeling van het totaal aantal advertenties ("total ads")
#*************************************************************************************************
# Als eerste stap vergelijken we de gemiddelde en verdeling van het aantal advertenties per gebruiker in de "ad" en "psa" groepen.
#*************************************************
# Gemiddeld aantal advertenties:
print("Ad group - Average total ads:", ad_group['total ads'].mean())
print("PSA group - Average total ads:", psa_group['total ads'].mean())

# Standart sapmalar
print("Ad group - Std total ads:", ad_group['total ads'].std())
print("PSA group - Std total ads:", psa_group['total ads'].std())
#************************************
"""
Observaties:

Het gemiddelde aantal advertenties is bijna gelijk.
De standaarddeviaties zijn ook behoorlijk vergelijkbaar.

➡ Dit geeft aan dat het systeem de gebruikers gelijk heeft verdeeld wat betreft het aantal getoonde advertenties.
Dit betekent dat er geen bias is tussen de groepen in termen van "hoeveel advertenties er worden getoond".

Wat betekent dit?
Met deze resultaten kunnen we concluderen dat de groepen gelijk zijn verdeeld volgens de "total ads" metric.
"""

#**********************************************************************
# 2. Verdeling van "Most Ads Day" (Meeste advertenties per dag)
#************************************************************
# Dagverdeling:
print("Most ads day (Ad groep):")
print(ad_group['most ads day'].value_counts(normalize=True))  # Zet aantallen om naar percentages
                                                            # (bijvoorbeeld als 100 uit 25 "Monday" is, is het resultaat 0.25).

print("\nMost ads day (PSA groep):")
print(psa_group['most ads day'].value_counts(normalize=True))


"""
Dagelijkse verdelingen zijn over het algemeen vergelijkbaar, maar er zijn enkele opvallende verschillen:
Donderdag (Thursday): De PSA-groep heeft duidelijk een hogere waarde (16,6% vs 14,01%)
Zondag (Sunday): De Ad-groep heeft een hogere waarde (14,58% vs 13,00%)
Op andere dagen zijn de verschillen ongeveer 1%, wat redelijk is.

Wat betekent dit?
Hoewel deze verschillen klein lijken, 
als het gebruikersgedrag op bepaalde dagen duidelijk verschilt, kan dit wijzen op veranderingen in aankoopgedrag,
vooral in het weekend of op werkdagen.
"""
#****************************************************************************
#   3- Uurverdeling
#     Most Ads Hour Verdeling
#**************************************
# Bereken de uurpercentages:
ad_hour_dist = ad_group['most ads hour'].value_counts(normalize=True).sort_index()  # De uren worden gesorteerd, dus van 0 tot 23.
psa_hour_dist = psa_group['most ads hour'].value_counts(normalize=True).sort_index()

# Maak een nieuwe DataFrame
hour_df = pd.DataFrame({
    'ad_group_ratio': ad_hour_dist,
    'psa_group_ratio': psa_hour_dist,
    'converted': df['converted']
}).fillna(0)  # Vul met 0 als er geen data is voor dat uur in een groep

# Voeg het uur als kolom toe:
hour_df.index.name = 'hour'
hour_df.reset_index(inplace=True)


"""
Prime time uren (19:00–23:00)
De Ad-groep wordt meestal meer vertegenwoordigd in deze uren. Het is bekend dat gebruikers gedurende deze tijdstippen actiever zijn op digitale apparaten en meer aankopen doen. Daarom kan het feit dat de Ad-groep in deze uren meer vertoningen ontvangt, de conversieratio positief beïnvloeden.

Resultaat:
Met deze analyse kunnen we het volgende duidelijk stellen:
De groepen "Ad" en "PSA" zijn gelijk verdeeld qua totaal aantal advertenties.
De dagelijkse verdelingen zijn vergelijkbaar, met kleine verschillen, maar geen dramatische afwijkingen.
De uurverdeling toont echter dat de Ad-groep 's avonds wat dominanter is, wat de conversieverschillen zou kunnen verklaren.
"""

#*********************************************************************
# 4- Conversieratio en Gebruikersaantal: Uurverdeling
# ******************************************************************
# Eerst filteren we alleen de gebruikers die "converted" zijn (d.w.z. die een aankoop hebben gedaan):

converted_df = df[df['converted'] == True]

# Tel het aantal conversies per uur (zonder normalisatie):
converted_hour_counts = converted_df['most ads hour'].value_counts().sort_index()

# Tel het aantal gebruikers per uur in totaal:
total_hour_counts = df['most ads hour'].value_counts().sort_index()

# Bereken de conversieratio per uur (converted / total):
conversion_rate_by_hour = (converted_hour_counts / total_hour_counts).fillna(0)

# Zet de data om naar een DataFrame:
hourly_conversion_df = pd.DataFrame({
    'conversion_rate': conversion_rate_by_hour,
    'total_users': total_hour_counts,
    'converted_users': converted_hour_counts
}).fillna(0)

# Voeg het uur als kolom toe:
hourly_conversion_df.index.name = 'hour'
hourly_conversion_df.reset_index(inplace=True)


# Visualisatie-- Conversion Rate by Hour
#*******************************************
plt.figure(figsize=(10, 6))
plt.plot(hourly_conversion_df['hour'], hourly_conversion_df['conversion_rate'], marker='o')
plt.title('Conversion Rate by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Conversion Rate')
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()
plt.show()

#****************************************************************
# 4-  Conversieratio op Basis van Tijdperken
#**************************************************************
# Bereken de conversieratio door de uren te groeperen

# Een functie om tijdsperiodes in te delen:
def get_time_period(hour):
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

# Nieuwe kolom maken: tijdsperiode
df['time_period'] = df['most ads hour'].apply(get_time_period)

# Conversies per tijdsperiode:
converted_df = df[df['converted'] == True]

# Aantal conversies per tijdsperiode
converted_counts_pertime = converted_df['time_period'].value_counts()
total_counts_pertime = df['time_period'].value_counts()

# Bereken de conversieratio
conversion_rate_by_period = (converted_counts_pertime / total_counts_pertime).fillna(0)

# Maak een DataFrame aan
period_df = pd.DataFrame({
    'conversion_rate': conversion_rate_by_period,
    'total_users': total_counts,
    'converted_users': converted_counts
}).fillna(0)

#Index hernoemen naar 'time_period'
period_df = period_df.reset_index().rename(columns={'index': 'time_period'}) # indexi yeniden sekillendiriyor.


# Visualisatie--Conversion Rate by Time of Day
#****************************************

plt.figure(figsize=(8, 5))
plt.bar(period_df['time_period'], period_df['conversion_rate'], color='coral', width=0.5)
plt.title('Conversion Rate by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Conversion Rate')
plt.ylim(0, period_df['conversion_rate'].max() * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
#***************************************************************************************

############################
# Beschrijvende Statistieken
############################

# Basisstatistieken voor de controlegroep
control_stats = ad_group.describe().T

# Basisstatistieken voor de testgroep
test_stats = psa_group.describe().T

# Statistieken combineren
summary_df = pd.concat([control_stats['mean'], test_stats['mean']], axis=1)
summary_df.columns = ['AD Group - Mean', 'PSA Group - Mean']


############################
# Confidence Intervals (Betrouwbaarheidsintervallen)
############################
sms.DescrStatsW(ad_group['converted']).tconfint_mean()
sms.DescrStatsW(psa_group['converted']).tconfint_mean()

######################################################
# Correlatie
######################################################
"""
Korelatiecoëfficiënt-Pearson (varieert tussen -1 en 1):
• Positieve correlatie
• Negatieve correlatie
• Geen correlatie

-1.00 - 0.00 → Negatieve correlatie
0.00 - 0.00 → Geen correlatie
0.00 - 1.00 → Positieve correlatie
"""
# Berekening van correlaties tussen variabelen

ad_group.head()

# Alleen Numerieke Kolommen Selecteren:
numeric_df = ad_group[['total ads', 'most ads hour']]
correlations = [ad_group['converted'].corr(ad_group[col]) for col in numeric_df.columns]

for col, corr in zip(numeric_df, correlations):
    print(f"{col} correlatie met converted: {corr}")

#Algemene Opmerking:
#**********************************************
"""
De positieve relatie tussen total ads en converted heeft enige betekenis, 
maar is niet erg sterk. Dat wil zeggen, het aantal advertenties lijkt de conversieratio een beetje te verhogen, maar dit effect is zwak.

De correlatie tussen most ads hour en converted is erg laag, 
wat aangeeft dat het tijdstip waarop de advertentie wordt weergegeven bijna geen invloed heeft op de conversie.

Deze resultaten geven aan dat het tijdstip van de advertenties de conversieratio niet significant beïnvloedt, 
en dat een toename van het aantal advertenties de conversieratio licht kan verhogen.

"""

# Correlatiefunctie:
#***************************************************************
def correlation_matrix(dataframe, target='converted'):
    numeric_df = dataframe[['total ads', 'most ads hour']]
    correlations = [dataframe[target].corr(dataframe[col]) for col in numeric_df.columns]
    for col, corr in zip(numeric_df, correlations):
        print(f"{col} correlatie met {target}: {corr}")

print('Voor de ad_group:')
correlation_matrix(ad_group)
print('Voor de psa_group:')
correlation_matrix(psa_group)

"""
Vergelijkende Resultaten:
Total ads → Dit is de meest gerelateerde variabele met conversie in beide groepen, maar de impact is zwak.

Most ads hour → Er is een zeer zwakke relatie in beide groepen. Het tijdstip lijkt geen belangrijke factor te zijn.

Ad group heeft over het algemeen een hogere correlatie, 
wat betekent dat de invloed van advertenties mogelijk iets groter is dan bij PSA.
"""
#********************************************************
"""
Beoordeling:
In beide groepen is de meest betekenisvolle correlatie met total ads. 
Dit betekent dat het tonen van meer advertenties de kans op conversie kan verhogen.

De correlatie met most ads hour is zeer zwak, 
wat betekent dat de dag van de week of het tijdsvenster nauwelijks invloed heeft op de conversie.

We zullen de conversieratio op basis van de total ads-intervals berekenen voor beide groepen (Ad en PSA):
Zodat we kunnen zien:
"Leidt het tonen van meer advertenties daadwerkelijk tot een hogere conversieratio?"

Wat we gaan doen:
--->> de total ads-waarden in bepaalde intervallen verdelen (bijvoorbeeld 0-100, 101-200, enz.).
--->> de conversieratio’s voor elk interval berekenen.
"""
#************************************************************
# Conversieratio Verdeling op Basis van Aantal Advertenties voor Ad en PSA Groepen:
#************************************************************
def conversion_by_ads_bin(df, group_name):
    # Deel het aantal advertenties op in bins (elk 100-interval)
    df['ads_bin'] = pd.cut(df['total ads'], bins=[0,100,200,300,400,500,1000], right=False)

    # Bereken de conversieratio voor elk interval
    conversion_rates = df.groupby('ads_bin')['converted'].mean().reset_index()
    conversion_rates.columns = ['ads_bin', 'conversion_rate']

    print(f"\n{group_name} Groep – Conversieratio op Basis van Aantal Advertenties:")
    print(conversion_rates)

    return conversion_rates

print("Ad Groep Conversieratio:")
ad_conversion = conversion_by_ads_bin(ad_group, "Ad")
print("\nPSA Groep Conversieratio:")
psa_conversion = conversion_by_ads_bin(psa_group, "PSA")

"""Algemene Beoordeling:
In beide groepen zien we dat gebruikers die meer advertenties zien, een hogere conversieratio hebben.

De Ad groep vertoont een meer consistente en sterke stijging.

De PSA groep vertoont ook een stijging, maar met enige fluctuatie."""

#  Visualisatie:
# -- Conversieratio op Basis van Aantal Advertenties
#*********************************
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(ad_conversion['ads_bin'].astype(str), ad_conversion['conversion_rate'], marker='o', label='Ad')
plt.plot(psa_conversion['ads_bin'].astype(str), psa_conversion['conversion_rate'], marker='o', label='PSA')
plt.xlabel("Aantal Advertenties (Bereik)")
plt.ylabel("Conversieratio")
plt.title("Conversieratio per Aantal Advertenties")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Analyse van conversieratio op basis van advertentiedag en -tijd
#*******************************************************************************

# Functie: Bereken de conversieratio’s per dag en uur voor een gegeven groep
def conversion_by_day_and_hour(dataframe, group_name):
    group_df = dataframe[dataframe['test group'] == group_name]

    # Conversieratio op basis van de dag waarop de meeste advertenties zijn getoond
    day_conv = group_df.groupby('most ads day')['converted'].mean().reset_index()
    day_conv.columns = ['most ads day', 'conversion_rate']

    # Conversieratio op basis van het uur waarop de meeste advertenties zijn getoond
    hour_conv = group_df.groupby('most ads hour')['converted'].mean().reset_index()
    hour_conv.columns = ['most ads hour', 'conversion_rate']

    return day_conv.sort_values('conversion_rate', ascending=False), hour_conv.sort_values('conversion_rate',
                                                                                           ascending=False)

# Analyses uitvoeren
ad_day, ad_hour = conversion_by_day_and_hour(df, 'ad')
psa_day, psa_hour = conversion_by_day_and_hour(df, 'psa')

# Resultaten:
print("🔹 Ad Groep – Conversieratio per Dag (van hoog naar laag)")
print(ad_day.sort_values('conversion_rate', ascending=False).to_string(index=False))

print("\n🔹 Ad Groep – Conversieratio per Uur (van hoog naar laag)")
print(ad_hour.sort_values('conversion_rate', ascending=False).to_string(index=False))

print("\n🔸 PSA Groep – Conversieratio per Dag (van hoog naar laag)")
print(psa_day.sort_values('conversion_rate', ascending=False).to_string(index=False))

print("\n🔸 PSA Groep – Conversieratio per Uur (van hoog naar laag)")
print(psa_hour.sort_values('conversion_rate', ascending=False).to_string(index=False))

# OPMERKING:
#***********************************************************
# Samenvattende Analyse – Conversieratio's

# --> Ad Groep
# Maandag en dinsdag vertonen een hogere conversieprestaties.
# Dit kunnen momenten zijn waarop mensen meer gefocust of geïnteresseerd zijn.

# Zaterdag en donderdag laten relatief lagere conversieratio's zien.
# Misschien komt dit door een vakantiegevoel of afleiding.

# Uurlijkse analyse:
# Conversieratio neemt aanzienlijk toe tussen 14:00 en 21:00.
# Deze tijdsperiode kan ideaal zijn voor het tonen van advertenties.
# Nachturen (0:00–5:00) hebben zeer lage conversie. Gebruikers zijn mogelijk minder actief.

# --> PSA Groep
# Net als bij de Ad groep, laat maandag de hoogste conversie zien.
# Zaterdag is het laagst – mogelijk minder aandacht van mensen.

# Uurlijkse piek: 16:00 → 2.81%
# PSA toont ook de hoogste prestaties rond 16:00. Dit kan het meest effectieve tijdstip van de dag zijn.

# Samenvattend:
# Zowel Ad als PSA presteren het beste op maandag en in de namiddag.
# Nachtelijke uren zijn minder effectief – misschien kan advertentie-uitgaven hier worden verminderd.
# Dag- en uurbasisstrategieën kunnen worden geoptimaliseerd voor verschillende advertentietypes.

#***********************************************

# 1. Conversieratio per Dag (Staafdiagram)

def plot_conversion_by_day(ad_day, psa_day):
    plt.figure(figsize=(10, 5))
    plt.bar(ad_day['most ads day'], ad_day['conversion_rate'], alpha=0.7, label='Ad', color='steelblue')
    plt.bar(psa_day['most ads day'], psa_day['conversion_rate'], alpha=0.7, label='PSA', color='orange')
    plt.xticks(range(7), ['Ma','Di','Wo','Do','Vr','Za','Zo'])
    plt.title('Conversieratio per Dag')
    plt.xlabel('Dag')
    plt.ylabel('Conversieratio')
    plt.legend()
    plt.grid(True)
    plt.show()

# 2. Conversieratio per Uur (Lijngrafiek)

def plot_conversion_by_hour(ad_hour, psa_hour):
    plt.figure(figsize=(12, 5))
    plt.plot(ad_hour['most ads hour'], ad_hour['conversion_rate'], marker='o', label='Ad', color='blue')
    plt.plot(psa_hour['most ads hour'], psa_hour['conversion_rate'], marker='o', label='PSA', color='orange')
    plt.title('Conversieratio per Uur')
    plt.xlabel('Uur')
    plt.ylabel('Conversieratio')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.legend()
    plt.show()


plot_conversion_by_day(ad_day, psa_day)
plot_conversion_by_hour(ad_hour, psa_hour)

#***********************************************
# Visuele correlatieanalyse:
#***********************************************

def correlation_matrix_heatmap(dataframe, target = 'converted'):
    numeric_df = dataframe[['total ads', 'most ads hour']]
    for col in numeric_df:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=dataframe, x='converted', y=col)
        plt.title(f'Converted vs {col}')
        plt.show()


correlation_matrix_heatmap(ad_group)
correlation_matrix_heatmap(psa_group)

#############################################
#   A/B TESTING
#############################################
# Hypothesevorming voor A/B-test
#############################################


# H0: M1 = M2 (AD en PSA vertonen geen significant verschil in 'converted' (aankoop) gemiddelden.)
# H1: M1 != M2 (...) er is een significant verschil.

#Bereken en vergelijk de gemiddelde aankoopcijfers (converted) van de AD-
# en PSA_group om een eerste indruk te krijgen van mogelijke verschillen.
#*******************************************************
df.head()
df.groupby("test group").agg({"converted": "mean",
                         "total ads": "mean",
                         "most ads hour": "mean"})

"""

1. conversiepercentage (converted)
* Voor de ad-groep: Gemiddeld conversiepercentage is 0.0255 (dus 2,55%).
* Voor de psa-groep: Gemiddeld conversiepercentage is 0.0179 (dus 1,79%).

Dit toont het conversiepercentage van elke groep. De ad-groep heeft een hoger conversiepercentage.

2. totaal aantal advertenties (total ads)
* Gemiddeld aantal advertenties is zeer vergelijkbaar: ongeveer 24.82 voor de ad-groep en 24.76 voor de psa-groep.
  Dit suggereert dat beide groepen aan een gelijkaardig aantal advertenties werden blootgesteld.

3. uur met de meeste advertenties (most ads hour)
* Voor de ad-groep: gemiddeld advertentie-uur is 14.4759, dus ongeveer 14:30.
* Voor de psa-groep: gemiddeld advertentie-uur is 14.3049, dus ongeveer 14:18.

Dit betekent dat advertenties in beide groepen vooral in de namiddag werden getoond.
Het tijdsverschil is klein, maar de ad-groep kreeg advertenties iets later te zien.

Samenvatting:
* Er is een duidelijk verschil in conversie: de ad-groep presteert beter.
* Het aantal advertenties is vrijwel gelijk tussen de groepen.
* Advertenties werden het vaakst rond 14:00 uur getoond.

Maar is dit verschil statistisch significant? Daarvoor moeten we de p-waarde berekenen.
Als p-waarde < 0,05, dan is het verschil significant en verwerpen we de nulhypothese H₀.
Als p-waarde ≥ 0,05, dan is het verschil waarschijnlijk toeval en behouden we H₀.
"""

#############################################
# Hypothesetest en Veronderstellingscontroles
#############################################
# 1 --->> Normaliteitstest
#           Beoordeel aan de hand van de testresultaten of de controle-
#           en testgroepen voldoen aan de normaliteitsaanname.
#************************************************************
# Hypothesecontrole
# H0: De ad_groep en de psa_groep voldoen aan de aanname van een normale verdeling in de 'converted'-gegevens.
# H1:..voldoet niet aan deze aanname.

missing_data = df.isnull().sum()

# Normale verdeling hypothesetest voor de controlegroep- ad_group (Shapiro)
test_stat, pvalue = shapiro(df.loc[df["test group"] == "ad", "converted"]) #.dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#-->> Test Stat = 0.1423, p-value = 0.0000
"""
BDe ad-groep bevat 564.577 observaties.
De Shapiro-Wilk test is krachtig bij kleine en middelgrote datasets.
Bij zeer grote datasets (N > 5000) kan deze test echter fout-positieve resultaten geven.

Daarom gebruiken we ook:
→ Kolmogorov-Smirnov (KS) test
"""

# Kolmogorov-Smirnov Test (voor grote datasets betrouwbaarder)
#******************************************************************************
test_stat, p_value = kstest(df.loc[df["test group"] == "ad", "converted"], 'norm')
print(f"KS Test Stat = {test_stat:.4f}, p-value = {p_value:.4f}")
#-->>KS Test Stat = 0.5000, p-value = 0.0000

"""
De ad-groep bevat 564.577 observaties.
De Shapiro-Wilk test is krachtig bij kleine en middelgrote datasets.
Bij zeer grote datasets (N > 5000) kan deze test echter fout-positieve resultaten geven.

Daarom gebruiken we ook:
→ Kolmogorov-Smirnov (KS) test
"""


# PSA-groep Normaliteitstoets
# Shapiro-Wilk test voor PSA-groep
test_stat, pvalue = shapiro(df.loc[df["test group"] == "psa", "converted"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Kolmogorov-Smirnov test
test_stat, p_value = kstest(df.loc[df["test group"] == "psa", "converted"], 'norm')
print(f"KS Test Stat = {test_stat:.4f}, p-value = {p_value:.4f}")


"""
Beide testen (Shapiro en KS) wijzen opnieuw op een niet-normale verdeling van de data.

→ p-waarde = 0.0000 → H0 wordt verworpen → geen normale verdeling.
→ Niet-parametrische tests zijn noodzakelijk.

"""


#2 --->> Homogeniteit van variantie (Levene’s Test)
#           Evalueer de testresultaten om te bepalen of
#           er een significant verschil is in variantie tussen de groepen.
#************************************************************
#Aanname van Variantiehomogeniteit
#H₀: De varianties zijn homogeen.
#H₁: De varianties zijn niet homogeen.

#Hypothesetest voor variantiehomogeniteit tussen de test- en controlegroep

test_stat, pvalue = levene(df.loc[df["test group"] == "ad", "converted"],
                                    df.loc[df["test group"] == "psa", "converted"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Als p-waarde < 0.05, dan wordt H₀ verworpen.
# Als p-waarde ≥ 0.05, dan kan H₀ niet worden verworpen.

"""
De Levene-test wordt gebruikt om te controleren of de varianties tussen verschillende groepen gelijk zijn.  
De nulhypothese (H0) van deze test stelt dat de varianties tussen de groepen gelijk zijn.  
De alternatieve hypothese (H1) stelt dat de varianties verschillend zijn.

Resultaten:
Teststatistiek (Test Stat): 54.3229  
p-waarde (p-value): 0.0000

Interpretatie:
Omdat de p-waarde 0.0000 is, verwerpen we de nulhypothese H0.  
Dit betekent dat de varianties tussen de groepen niet gelijk zijn.

De teststatistiek van de Levene-test is vrij hoog (54.3229),  
wat bevestigt dat er een aanzienlijk verschil is in variantie tussen de groepen.

Conclusie:
Volgens deze test zijn de varianties van de conversieratio's in de "ad"-groep en de "psa"-groep niet gelijk.  
Met andere woorden, het niveau van variabiliteit verschilt tussen de twee groepen.

Aanbeveling:
Als blijkt dat de varianties ongelijk zijn, moeten we bij het gebruik van parametrische testen de aanname van gelijke variantie negeren.  
In plaats daarvan kunnen we beter niet-parametrische testen gebruiken, of testen die geschikt zijn voor ongelijke varianties, zoals de Welch’s t-test.
"""


###################################################################################################################################
#  Mann-Whitney U-test
# #************************************************************

test_stat, pvalue = mannwhitneyu(df.loc[df["test group"] == "ad", "converted"],
                                 df.loc[df["test group"] == "psa", "converted"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))




means = df.groupby("test group")["converted"].mean()
print(means)

"""
De gemiddelde conversieratio van de Ad-groep: 2,55%  
De gemiddelde conversieratio van de PSA-groep: 1,79%

Met andere woorden, de conversieratio is hoger in de Ad-groep.  
De Mann-Whitney-test toont ook aan dat dit verschil statistisch significant is.
"""

################################################################################################################################
# Conclusie en Aanbeveling
#############################################
"""
Doel van de test:
De Mann-Whitney U-test test of de mediaan van twee onafhankelijke groepen (bijvoorbeeld 'ad' en 'psa') verschillend is.
Deze test wordt vooral gebruikt wanneer de gegevens niet normaal verdeeld zijn of binair (bijvoorbeeld 0 of 1) zijn — wat het een zeer geschikte test maakt voor deze dataset.

Testresultaat:
p-waarde = 0,0000, dus p < 0,05
→ Dit duidt op een statistisch significant verschil.
→ Er is een significant verschil tussen de conversiepercentages van de 'ad' en 'psa' groepen.

H0 (nulhypothese): Er is geen verschil tussen de conversiepercentages van de 'ad' en 'psa' groepen.
Resultaat: Omdat p < 0,05, wordt H0 verworpen.
→ Dit betekent dat er een significant verschil is tussen de conversiesucces tussen de twee groepen.
"""

#  --->> Op basis van de testresultaten, welke biedstrategie zou aanbevelen?
#***********************************************

"""

1. Was de reclamecampagne succesvol?
Ja. De Ad-groep had een conversieratio van 2,55%, PSA slechts 1,79%.
→ Dit verschil is statistisch significant (p < 0.05).

2. In hoeverre is het succes toe te schrijven aan advertenties?
→ Aangezien de enige variabele het advertentietype was (randomisatie werd gehanteerd),
kan het verschil direct aan advertenties worden toegeschreven.

Bijkomende observaties:
- Tijdsgebonden patronen: Weekdagen & namiddag zijn effectiever.
- Meer advertenties tonen heeft een lichte positieve invloed.

Conclusie:
Conversiestijging is grotendeels te danken aan advertenties.


2. In hoeverre is het succes toe te schrijven aan de advertenties?
Het effect van de advertenties werd direct getest. In de A/B-test:

De 'ad' groep werd blootgesteld aan advertenties.

De 'psa' groep zag geen advertenties.

De enige verschil tussen de twee groepen was of ze advertenties zagen of niet, alle andere factoren waren gelijk (aangenomen dat er een willekeurige toewijzing was).
Daarom kan het verschil in conversiepercentages worden toegeschreven aan het effect van de advertenties.

Daarnaast:

Advertentietijden en dagen kwamen naar voren als factoren die de conversie beïnvloedden (bijvoorbeeld doordeweeks en in de middag waren succesvoller).
Er was een zwakke, maar positieve relatie tussen het aantal vertoningen (totale advertenties) en conversie.

Conclusie: Het succes in de conversie is grotendeels afhankelijk van de advertenties.

*******************************************
Stratejik Öneriler – Pazarlama İçin
*******************************************
Investeer meer in advertenties: De Ad-groep converteerde significant beter dan de PSA-groep.

Optimaliseer timing:
→ Focus op werkdagen (vooral maandag).
→ Richt advertenties op de namiddag (14:00–17:00).

Verminder advertenties in het weekend (bijv. zaterdag) – lage prestaties.

Beperk overmatige herhaling: Positieve correlatie met conversie, maar vermijd irritatie.

Segmenteer verder:
→ Analyseer prestaties per demografische groep of apparaattype voor betere targeting.
"""