\* Allocation_Azote_Animaux *\
Minimize
OBJ: 0.5 Bioraffinery_energy_dev
 + 0.000349267576201 I_('Barley_grain',_'French')
 + 0.000349267576201 I_('Barley_grain',_'bovines')
 + 0.000349267576201 I_('Barley_grain',_'porcines')
 + 0.000349267576201 I_('Forage',_'bovines')
 + 0.000349267576201 I_('Maize_corn',_'French')
 + 0.000349267576201 I_('Maize_corn',_'porcines')
 + 0.000349267576201 I_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 + 0.000349267576201 I_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 + 0.000349267576201 I_('Natural_meadow_forage',_'bovines')
 + 0.000349267576201 I_('Oats_grain',_'French')
 + 0.000349267576201 I_('Raw_milk_of_cattle',_'French')
 + 0.000349267576201 I_('Rice_grain',_'French')
 + 0.000349267576201 I_('Soya_beans_grain',_'French')
 + 0.000349267576201 I_('Soya_beans_grain',_'bovines')
 + 0.000349267576201 I_('Soya_beans_grain',_'porcines')
 + 0.000349267576201 I_('Wheat_grain',_'French')
 + 0.000349267576201 I_('Wheat_grain',_'bovines')
 + 0.000349267576201 I_('Wheat_grain',_'porcines')
 + 0.000349267576201 I_energy_Forage_Bioraffinery
 + 0.000349267576201 I_energy_Wheat_grain_Bioraffinery
 + 0.5 Methanizer_energy_dev + 0.0625 delta_('French',_('Barley_grain',))
 + 0.0625 delta_('French',_('Maize_corn',))
 + 0.0625 delta_('French',_('Meat_of_cattle_with_the_bone_fresh_or_chilled',))
 + 0.0625 delta_('French',_('Meat_of_pig_with_the_bone_fresh_or_chilled',))
 + 0.0625 delta_('French',_('Oats_grain',))
 + 0.0625 delta_('French',_('Raw_milk_of_cattle',))
 + 0.0625 delta_('French',_('Rice_grain',))
 + 0.0625 delta_('French',_('Soya_beans_grain',))
 + 0.0625 delta_('French',_('Wheat_grain',))
 + 0.0625 delta_('bovines',_('Barley_grain',_'Wheat_grain'))
 + 0.0625 delta_('bovines',_('Natural_meadow_forage',_'Forage'))
 + 0.0625 delta_('bovines',_('Soya_beans_grain',))
 + 0.0625 delta_('porcines',_('Barley_grain',))
 + 0.0625 delta_('porcines',_('Maize_corn',))
 + 0.0625 delta_('porcines',_('Soya_beans_grain',))
 + 0.0625 delta_('porcines',_('Wheat_grain',))
 + 1.57828282828e-05 gamma_fair_abs_('Barley_grain',_'French')
 + 1.57828282828e-05 gamma_fair_abs_('Barley_grain',_'bovines')
 + 1.57828282828e-05 gamma_fair_abs_('Barley_grain',_'porcines')
 + 0.00416666666667 gamma_fair_abs_('Forage',_'bovines')
 + 5.55555555556e-05 gamma_fair_abs_('Maize_corn',_'French')
 + 5.55555555556e-05 gamma_fair_abs_('Maize_corn',_'porcines')
 + 0.000126763534289 gamma_fair_abs_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 + 0.000123932691659 gamma_fair_abs_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 + 2.0325203252e-05 gamma_fair_abs_('Natural_meadow_forage',_'bovines')
 + 7.73232277195e-05 gamma_fair_abs_('Oats_grain',_'French')
 + 4.12266673092e-05 gamma_fair_abs_('Raw_milk_of_cattle',_'French')
 + 0.00291276909035 gamma_fair_abs_('Rice_grain',_'French')
 + 0.000115740740741 gamma_fair_abs_('Soya_beans_grain',_'French')
 + 0.000115740740741 gamma_fair_abs_('Soya_beans_grain',_'bovines')
 + 0.000115740740741 gamma_fair_abs_('Soya_beans_grain',_'porcines')
 + 8.54700854701e-06 gamma_fair_abs_('Wheat_grain',_'French')
 + 8.54700854701e-06 gamma_fair_abs_('Wheat_grain',_'bovines')
 + 8.54700854701e-06 gamma_fair_abs_('Wheat_grain',_'porcines')
 + 0.00416666666667 gamma_fair_abs_Bioraffinery_Forage
 + 8.54700854701e-06 gamma_fair_abs_Bioraffinery_Wheat_grain
 + 1.57828282828e-05 gamma_fair_abs_Methanizer_Barley_grain
 + 0.00416666666667 gamma_fair_abs_Methanizer_Forage
 + 5.55555555556e-05 gamma_fair_abs_Methanizer_Maize_corn
 + 2.0325203252e-05 gamma_fair_abs_Methanizer_Natural_meadow_forage
 + 7.73232277195e-05 gamma_fair_abs_Methanizer_Oats_grain
 + 8.54700854701e-06 gamma_fair_abs_Methanizer_Wheat_grain
 + 0.00833333333333 penalite_culture_('French',_0.025,_'Barley_grain')
 + 0.00833333333333 penalite_culture_('French',_0.025,_'Oats_grain')
 + 0.00833333333333 penalite_culture_('French',_0.05,_'Maize_corn')
 + 0.00833333333333 penalite_culture_('French',_0.05,_'Raw_milk_of_cattle')
 + 0.00833333333333 penalite_culture_('French',_0.05,_'Rice_grain')
 + 0.00833333333333 penalite_culture_('French',_0.05,_'Soya_beans_grain')
 + 0.00833333333333 penalite_culture_('French',_0.15,_'Meat_of_pig_with_the_bone_fresh_or_chilled')
 + 0.00833333333333 penalite_culture_('French',_0.3,_'Meat_of_cattle_with_the_bone_fresh_or_chilled')
 + 0.00833333333333 penalite_culture_('French',_0.3,_'Wheat_grain')
 + 0.00833333333333 penalite_culture_('bovines',_0.2,_'Soya_beans_grain')
 + 0.00833333333333 penalite_culture_('bovines',_0.3,_'Barley_grain')
 + 0.00833333333333 penalite_culture_('bovines',_0.3,_'Wheat_grain')
 + 0.00833333333333 penalite_culture_('bovines',_0.5,_'Forage')
 + 0.00833333333333 penalite_culture_('bovines',_0.5,_'Natural_meadow_forage')
 + 0.00833333333333 penalite_culture_('porcines',_0.1,_'Maize_corn')
 + 0.00833333333333 penalite_culture_('porcines',_0.2,_'Barley_grain')
 + 0.00833333333333 penalite_culture_('porcines',_0.3,_'Wheat_grain')
 + 0.00833333333333 penalite_culture_('porcines',_0.4,_'Soya_beans_grain')
 + 0.000416666666667 penalite_culture_Bioraffinery_('Bioraffinery',_0.5,_'Forage')
 + 0.000416666666667 penalite_culture_Bioraffinery_('Bioraffinery',_0.5,_'Wheat_grain')
 + 7.57575757576e-05 penalite_culture_Methanizer_('Methanizer',_0.1,_'Forage')
 + 7.57575757576e-05 penalite_culture_Methanizer_('Methanizer',_0.1,_'Natural_meadow_forage')
 + 7.57575757576e-05 penalite_culture_Methanizer_('Methanizer',_0.1,_'waste')
 + 7.57575757576e-05 penalite_culture_Methanizer_('Methanizer',_0.2,_'Barley_grain')
 + 7.57575757576e-05 penalite_culture_Methanizer_('Methanizer',_0.2,_'Maize_corn')
 + 7.57575757576e-05 penalite_culture_Methanizer_('Methanizer',_0.2,_'Oats_grain')
 + 7.57575757576e-05 penalite_culture_Methanizer_('Methanizer',_0.2,_'Wheat_grain')
 + 7.57575757576e-05 penalite_culture_Methanizer_('Methanizer',_0.6,_'bovines_manure')
 + 7.57575757576e-05 penalite_culture_Methanizer_('Methanizer',_0.6,_'bovines_slurry')
 + 7.57575757576e-05 penalite_culture_Methanizer_('Methanizer',_0.6,_'porcines_manure')
 + 7.57575757576e-05 penalite_culture_Methanizer_('Methanizer',_0.6,_'porcines_slurry')

Subject To
Besoin_French: I_('Barley_grain',_'French') + I_('Maize_corn',_'French')
 + I_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 + I_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 + I_('Oats_grain',_'French') + I_('Raw_milk_of_cattle',_'French')
 + I_('Rice_grain',_'French') + I_('Soya_beans_grain',_'French')
 + I_('Wheat_grain',_'French') + x_('Barley_grain',_'French')
 + x_('Maize_corn',_'French')
 + x_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 + x_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 + x_('Oats_grain',_'French') + x_('Raw_milk_of_cattle',_'French')
 + x_('Rice_grain',_'French') + x_('Soya_beans_grain',_'French')
 + x_('Wheat_grain',_'French') = 200
Besoin_bovines: I_('Barley_grain',_'bovines') + I_('Forage',_'bovines')
 + I_('Natural_meadow_forage',_'bovines') + I_('Soya_beans_grain',_'bovines')
 + I_('Wheat_grain',_'bovines') + x_('Barley_grain',_'bovines')
 + x_('Forage',_'bovines') + x_('Natural_meadow_forage',_'bovines')
 + x_('Soya_beans_grain',_'bovines') + x_('Wheat_grain',_'bovines')
 = 3059.0880688
Besoin_porcines: I_('Barley_grain',_'porcines') + I_('Maize_corn',_'porcines')
 + I_('Soya_beans_grain',_'porcines') + I_('Wheat_grain',_'porcines')
 + x_('Barley_grain',_'porcines') + x_('Maize_corn',_'porcines')
 + x_('Soya_beans_grain',_'porcines') + x_('Wheat_grain',_'porcines')
 = 894.4204
Bilan_Barley_grain: U_Barley_grain + x_('Barley_grain',_'French')
 + x_('Barley_grain',_'bovines') + x_('Barley_grain',_'porcines')
 + x_Methanizer_prod_Barley_grain = 250.8
Bilan_Forage: U_Forage + x_('Forage',_'bovines') + x_Bioraffinery_prod_Forage
 + x_Methanizer_prod_Forage = 0
Bilan_Maize_corn: U_Maize_corn + x_('Maize_corn',_'French')
 + x_('Maize_corn',_'porcines') + x_Methanizer_prod_Maize_corn = 71.25
Bilan_Meat_of_cattle_with_the_bone_fresh_or_chilled:
 U_Meat_of_cattle_with_the_bone_fresh_or_chilled
 + x_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French') = 31.22612
Bilan_Meat_of_pig_with_the_bone_fresh_or_chilled:
 U_Meat_of_pig_with_the_bone_fresh_or_chilled
 + x_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French') = 31.93938
Bilan_Natural_meadow_forage: U_Natural_meadow_forage
 + x_('Natural_meadow_forage',_'bovines')
 + x_Methanizer_prod_Natural_meadow_forage = 190.65
Bilan_Non_edible_porcines_carcass: U_Non_edible_porcines_carcass = 0.475
Bilan_Oats_grain: U_Oats_grain + x_('Oats_grain',_'French')
 + x_Methanizer_prod_Oats_grain = 51.1920344
Bilan_Raw_milk_of_cattle: U_Raw_milk_of_cattle
 + x_('Raw_milk_of_cattle',_'French') = 96.01390536
Bilan_Rice_grain: U_Rice_grain + x_('Rice_grain',_'French') = 1.35895885
Bilan_Soya_beans_grain: U_Soya_beans_grain + x_('Soya_beans_grain',_'French')
 + x_('Soya_beans_grain',_'bovines') + x_('Soya_beans_grain',_'porcines')
 = 34.2
Bilan_Wheat_grain: U_Wheat_grain + x_('Wheat_grain',_'French')
 + x_('Wheat_grain',_'bovines') + x_('Wheat_grain',_'porcines')
 + x_Bioraffinery_prod_Wheat_grain + x_Methanizer_prod_Wheat_grain = 463.125
Contrainte_part_animale_French:
 I_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 + I_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 + I_('Raw_milk_of_cattle',_'French')
 + x_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 + x_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 + x_('Raw_milk_of_cattle',_'French') = 100
_dummy: __dummy = 0
Contrainte_part_animale_bovines: __dummy = 0
Contrainte_part_animale_porcines: __dummy = 0
Deviation_Moins_French_('Barley_grain',): - 0.005 I_('Barley_grain',_'French')
 - delta_('French',_('Barley_grain',)) - 0.005 x_('Barley_grain',_'French')
 <= -0.025
Deviation_Moins_French_('Maize_corn',): - 0.005 I_('Maize_corn',_'French')
 - delta_('French',_('Maize_corn',)) - 0.005 x_('Maize_corn',_'French')
 <= -0.05
Deviation_Moins_French_('Meat_of_cattle_with_the_bone_fresh_or_chilled',):
 - 0.005 I_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 - delta_('French',_('Meat_of_cattle_with_the_bone_fresh_or_chilled',))
 - 0.005 x_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French') <= -0.3
Deviation_Moins_French_('Meat_of_pig_with_the_bone_fresh_or_chilled',):
 - 0.005 I_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 - delta_('French',_('Meat_of_pig_with_the_bone_fresh_or_chilled',))
 - 0.005 x_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French') <= -0.15
Deviation_Moins_French_('Oats_grain',): - 0.005 I_('Oats_grain',_'French')
 - delta_('French',_('Oats_grain',)) - 0.005 x_('Oats_grain',_'French')
 <= -0.025
Deviation_Moins_French_('Raw_milk_of_cattle',):
 - 0.005 I_('Raw_milk_of_cattle',_'French')
 - delta_('French',_('Raw_milk_of_cattle',))
 - 0.005 x_('Raw_milk_of_cattle',_'French') <= -0.05
Deviation_Moins_French_('Rice_grain',): - 0.005 I_('Rice_grain',_'French')
 - delta_('French',_('Rice_grain',)) - 0.005 x_('Rice_grain',_'French')
 <= -0.05
Deviation_Moins_French_('Soya_beans_grain',):
 - 0.005 I_('Soya_beans_grain',_'French')
 - delta_('French',_('Soya_beans_grain',))
 - 0.005 x_('Soya_beans_grain',_'French') <= -0.05
Deviation_Moins_French_('Wheat_grain',): - 0.005 I_('Wheat_grain',_'French')
 - delta_('French',_('Wheat_grain',)) - 0.005 x_('Wheat_grain',_'French')
 <= -0.3
Deviation_Moins_bovines_('Barley_grain',_'Wheat_grain'):
 - 0.000326894805743 I_('Barley_grain',_'bovines')
 - 0.000326894805743 I_('Wheat_grain',_'bovines')
 - delta_('bovines',_('Barley_grain',_'Wheat_grain'))
 - 0.000326894805743 x_('Barley_grain',_'bovines')
 - 0.000326894805743 x_('Wheat_grain',_'bovines') <= -0.3
Deviation_Moins_bovines_('Natural_meadow_forage',_'Forage'):
 - 0.000326894805743 I_('Forage',_'bovines')
 - 0.000326894805743 I_('Natural_meadow_forage',_'bovines')
 - delta_('bovines',_('Natural_meadow_forage',_'Forage'))
 - 0.000326894805743 x_('Forage',_'bovines')
 - 0.000326894805743 x_('Natural_meadow_forage',_'bovines') <= -0.5
Deviation_Moins_bovines_('Soya_beans_grain',):
 - 0.000326894805743 I_('Soya_beans_grain',_'bovines')
 - delta_('bovines',_('Soya_beans_grain',))
 - 0.000326894805743 x_('Soya_beans_grain',_'bovines') <= -0.2
Deviation_Moins_porcines_('Barley_grain',):
 - 0.00111804247756 I_('Barley_grain',_'porcines')
 - delta_('porcines',_('Barley_grain',))
 - 0.00111804247756 x_('Barley_grain',_'porcines') <= -0.2
Deviation_Moins_porcines_('Maize_corn',):
 - 0.00111804247756 I_('Maize_corn',_'porcines')
 - delta_('porcines',_('Maize_corn',))
 - 0.00111804247756 x_('Maize_corn',_'porcines') <= -0.1
Deviation_Moins_porcines_('Soya_beans_grain',):
 - 0.00111804247756 I_('Soya_beans_grain',_'porcines')
 - delta_('porcines',_('Soya_beans_grain',))
 - 0.00111804247756 x_('Soya_beans_grain',_'porcines') <= -0.4
Deviation_Moins_porcines_('Wheat_grain',):
 - 0.00111804247756 I_('Wheat_grain',_'porcines')
 - delta_('porcines',_('Wheat_grain',))
 - 0.00111804247756 x_('Wheat_grain',_'porcines') <= -0.3
Deviation_Plus_French_('Barley_grain',): 0.005 I_('Barley_grain',_'French')
 - delta_('French',_('Barley_grain',)) + 0.005 x_('Barley_grain',_'French')
 <= 0.025
Deviation_Plus_French_('Maize_corn',): 0.005 I_('Maize_corn',_'French')
 - delta_('French',_('Maize_corn',)) + 0.005 x_('Maize_corn',_'French')
 <= 0.05
Deviation_Plus_French_('Meat_of_cattle_with_the_bone_fresh_or_chilled',):
 0.005 I_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 - delta_('French',_('Meat_of_cattle_with_the_bone_fresh_or_chilled',))
 + 0.005 x_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French') <= 0.3
Deviation_Plus_French_('Meat_of_pig_with_the_bone_fresh_or_chilled',):
 0.005 I_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 - delta_('French',_('Meat_of_pig_with_the_bone_fresh_or_chilled',))
 + 0.005 x_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French') <= 0.15
Deviation_Plus_French_('Oats_grain',): 0.005 I_('Oats_grain',_'French')
 - delta_('French',_('Oats_grain',)) + 0.005 x_('Oats_grain',_'French')
 <= 0.025
Deviation_Plus_French_('Raw_milk_of_cattle',):
 0.005 I_('Raw_milk_of_cattle',_'French')
 - delta_('French',_('Raw_milk_of_cattle',))
 + 0.005 x_('Raw_milk_of_cattle',_'French') <= 0.05
Deviation_Plus_French_('Rice_grain',): 0.005 I_('Rice_grain',_'French')
 - delta_('French',_('Rice_grain',)) + 0.005 x_('Rice_grain',_'French')
 <= 0.05
Deviation_Plus_French_('Soya_beans_grain',):
 0.005 I_('Soya_beans_grain',_'French')
 - delta_('French',_('Soya_beans_grain',))
 + 0.005 x_('Soya_beans_grain',_'French') <= 0.05
Deviation_Plus_French_('Wheat_grain',): 0.005 I_('Wheat_grain',_'French')
 - delta_('French',_('Wheat_grain',)) + 0.005 x_('Wheat_grain',_'French')
 <= 0.3
Deviation_Plus_bovines_('Barley_grain',_'Wheat_grain'):
 0.000326894805743 I_('Barley_grain',_'bovines')
 + 0.000326894805743 I_('Wheat_grain',_'bovines')
 - delta_('bovines',_('Barley_grain',_'Wheat_grain'))
 + 0.000326894805743 x_('Barley_grain',_'bovines')
 + 0.000326894805743 x_('Wheat_grain',_'bovines') <= 0.3
Deviation_Plus_bovines_('Natural_meadow_forage',_'Forage'):
 0.000326894805743 I_('Forage',_'bovines')
 + 0.000326894805743 I_('Natural_meadow_forage',_'bovines')
 - delta_('bovines',_('Natural_meadow_forage',_'Forage'))
 + 0.000326894805743 x_('Forage',_'bovines')
 + 0.000326894805743 x_('Natural_meadow_forage',_'bovines') <= 0.5
Deviation_Plus_bovines_('Soya_beans_grain',):
 0.000326894805743 I_('Soya_beans_grain',_'bovines')
 - delta_('bovines',_('Soya_beans_grain',))
 + 0.000326894805743 x_('Soya_beans_grain',_'bovines') <= 0.2
Deviation_Plus_porcines_('Barley_grain',):
 0.00111804247756 I_('Barley_grain',_'porcines')
 - delta_('porcines',_('Barley_grain',))
 + 0.00111804247756 x_('Barley_grain',_'porcines') <= 0.2
Deviation_Plus_porcines_('Maize_corn',):
 0.00111804247756 I_('Maize_corn',_'porcines')
 - delta_('porcines',_('Maize_corn',))
 + 0.00111804247756 x_('Maize_corn',_'porcines') <= 0.1
Deviation_Plus_porcines_('Soya_beans_grain',):
 0.00111804247756 I_('Soya_beans_grain',_'porcines')
 - delta_('porcines',_('Soya_beans_grain',))
 + 0.00111804247756 x_('Soya_beans_grain',_'porcines') <= 0.4
Deviation_Plus_porcines_('Wheat_grain',):
 0.00111804247756 I_('Wheat_grain',_'porcines')
 - delta_('porcines',_('Wheat_grain',))
 + 0.00111804247756 x_('Wheat_grain',_'porcines') <= 0.3
Excreta_cap_bovines_manure: x_Methanizer_excr_bovines_manure <= 342.696673866
Excreta_cap_bovines_slurry: x_Methanizer_excr_bovines_slurry <= 83.2439528496
Excreta_cap_porcines_manure: x_Methanizer_excr_porcines_manure <= 17.2885888
Excreta_cap_porcines_slurry: x_Methanizer_excr_porcines_slurry <= 173.43648
FairAbs_Bioraffinery_moins_Forage: - gamma_fair_abs_Bioraffinery_Forage
 - x_Bioraffinery_prod_Forage <= 0
FairAbs_Bioraffinery_moins_Wheat_grain:
 - gamma_fair_abs_Bioraffinery_Wheat_grain - x_Bioraffinery_prod_Wheat_grain
 <= -195
FairAbs_Bioraffinery_plus_Forage: - gamma_fair_abs_Bioraffinery_Forage
 + x_Bioraffinery_prod_Forage <= 0
FairAbs_Bioraffinery_plus_Wheat_grain:
 - gamma_fair_abs_Bioraffinery_Wheat_grain + x_Bioraffinery_prod_Wheat_grain
 <= 195
FairAbs_Methanizer_moins_Barley_grain:
 - gamma_fair_abs_Methanizer_Barley_grain - x_Methanizer_prod_Barley_grain
 <= -31.0588235294
FairAbs_Methanizer_moins_Forage: - gamma_fair_abs_Methanizer_Forage
 - x_Methanizer_prod_Forage <= 0
FairAbs_Methanizer_moins_Maize_corn: - gamma_fair_abs_Methanizer_Maize_corn
 - x_Methanizer_prod_Maize_corn <= -18.75
FairAbs_Methanizer_moins_Natural_meadow_forage:
 - gamma_fair_abs_Methanizer_Natural_meadow_forage
 - x_Methanizer_prod_Natural_meadow_forage <= -34.1666666667
FairAbs_Methanizer_moins_Oats_grain: - gamma_fair_abs_Methanizer_Oats_grain
 - x_Methanizer_prod_Oats_grain <= -35.9242346667
FairAbs_Methanizer_moins_Wheat_grain: - gamma_fair_abs_Methanizer_Wheat_grain
 - x_Methanizer_prod_Wheat_grain <= -30.46875
FairAbs_Methanizer_plus_Barley_grain: - gamma_fair_abs_Methanizer_Barley_grain
 + x_Methanizer_prod_Barley_grain <= 31.0588235294
FairAbs_Methanizer_plus_Forage: - gamma_fair_abs_Methanizer_Forage
 + x_Methanizer_prod_Forage <= 0
FairAbs_Methanizer_plus_Maize_corn: - gamma_fair_abs_Methanizer_Maize_corn
 + x_Methanizer_prod_Maize_corn <= 18.75
FairAbs_Methanizer_plus_Natural_meadow_forage:
 - gamma_fair_abs_Methanizer_Natural_meadow_forage
 + x_Methanizer_prod_Natural_meadow_forage <= 34.1666666667
FairAbs_Methanizer_plus_Oats_grain: - gamma_fair_abs_Methanizer_Oats_grain
 + x_Methanizer_prod_Oats_grain <= 35.9242346667
FairAbs_Methanizer_plus_Wheat_grain: - gamma_fair_abs_Methanizer_Wheat_grain
 + x_Methanizer_prod_Wheat_grain <= 30.46875
FairAbs_moins_Barley_grain_French: - gamma_fair_abs_('Barley_grain',_'French')
 - x_('Barley_grain',_'French') <= -17.6
FairAbs_moins_Barley_grain_bovines:
 - gamma_fair_abs_('Barley_grain',_'bovines') - x_('Barley_grain',_'bovines')
 <= -105.6
FairAbs_moins_Barley_grain_porcines:
 - gamma_fair_abs_('Barley_grain',_'porcines')
 - x_('Barley_grain',_'porcines') <= -140.8
FairAbs_moins_Forage_bovines: - gamma_fair_abs_('Forage',_'bovines')
 - x_('Forage',_'bovines') <= 0
FairAbs_moins_Maize_corn_French: - gamma_fair_abs_('Maize_corn',_'French')
 - x_('Maize_corn',_'French') <= -25
FairAbs_moins_Maize_corn_porcines: - gamma_fair_abs_('Maize_corn',_'porcines')
 - x_('Maize_corn',_'porcines') <= -50
FairAbs_moins_Meat_of_cattle_with_the_bone_fresh_or_chilled_French:
 - gamma_fair_abs_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 - x_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French') <= -32.8696
FairAbs_moins_Meat_of_pig_with_the_bone_fresh_or_chilled_French:
 - gamma_fair_abs_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 - x_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French') <= -33.6204
FairAbs_moins_Natural_meadow_forage_bovines:
 - gamma_fair_abs_('Natural_meadow_forage',_'bovines')
 - x_('Natural_meadow_forage',_'bovines') <= -205
FairAbs_moins_Oats_grain_French: - gamma_fair_abs_('Oats_grain',_'French')
 - x_('Oats_grain',_'French') <= -53.886352
FairAbs_moins_Raw_milk_of_cattle_French:
 - gamma_fair_abs_('Raw_milk_of_cattle',_'French')
 - x_('Raw_milk_of_cattle',_'French') <= -101.0672688
FairAbs_moins_Rice_grain_French: - gamma_fair_abs_('Rice_grain',_'French')
 - x_('Rice_grain',_'French') <= -1.430483
FairAbs_moins_Soya_beans_grain_French:
 - gamma_fair_abs_('Soya_beans_grain',_'French')
 - x_('Soya_beans_grain',_'French') <= -2.76923076923
FairAbs_moins_Soya_beans_grain_bovines:
 - gamma_fair_abs_('Soya_beans_grain',_'bovines')
 - x_('Soya_beans_grain',_'bovines') <= -11.0769230769
FairAbs_moins_Soya_beans_grain_porcines:
 - gamma_fair_abs_('Soya_beans_grain',_'porcines')
 - x_('Soya_beans_grain',_'porcines') <= -22.1538461538
FairAbs_moins_Wheat_grain_French: - gamma_fair_abs_('Wheat_grain',_'French')
 - x_('Wheat_grain',_'French') <= -195
FairAbs_moins_Wheat_grain_bovines: - gamma_fair_abs_('Wheat_grain',_'bovines')
 - x_('Wheat_grain',_'bovines') <= -97.5
FairAbs_moins_Wheat_grain_porcines:
 - gamma_fair_abs_('Wheat_grain',_'porcines') - x_('Wheat_grain',_'porcines')
 <= -195
FairAbs_plus_Barley_grain_French: - gamma_fair_abs_('Barley_grain',_'French')
 + x_('Barley_grain',_'French') <= 17.6
FairAbs_plus_Barley_grain_bovines:
 - gamma_fair_abs_('Barley_grain',_'bovines') + x_('Barley_grain',_'bovines')
 <= 105.6
FairAbs_plus_Barley_grain_porcines:
 - gamma_fair_abs_('Barley_grain',_'porcines')
 + x_('Barley_grain',_'porcines') <= 140.8
FairAbs_plus_Forage_bovines: - gamma_fair_abs_('Forage',_'bovines')
 + x_('Forage',_'bovines') <= 0
FairAbs_plus_Maize_corn_French: - gamma_fair_abs_('Maize_corn',_'French')
 + x_('Maize_corn',_'French') <= 25
FairAbs_plus_Maize_corn_porcines: - gamma_fair_abs_('Maize_corn',_'porcines')
 + x_('Maize_corn',_'porcines') <= 50
FairAbs_plus_Meat_of_cattle_with_the_bone_fresh_or_chilled_French:
 - gamma_fair_abs_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 + x_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French') <= 32.8696
FairAbs_plus_Meat_of_pig_with_the_bone_fresh_or_chilled_French:
 - gamma_fair_abs_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 + x_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French') <= 33.6204
FairAbs_plus_Natural_meadow_forage_bovines:
 - gamma_fair_abs_('Natural_meadow_forage',_'bovines')
 + x_('Natural_meadow_forage',_'bovines') <= 205
FairAbs_plus_Oats_grain_French: - gamma_fair_abs_('Oats_grain',_'French')
 + x_('Oats_grain',_'French') <= 53.886352
FairAbs_plus_Raw_milk_of_cattle_French:
 - gamma_fair_abs_('Raw_milk_of_cattle',_'French')
 + x_('Raw_milk_of_cattle',_'French') <= 101.0672688
FairAbs_plus_Rice_grain_French: - gamma_fair_abs_('Rice_grain',_'French')
 + x_('Rice_grain',_'French') <= 1.430483
FairAbs_plus_Soya_beans_grain_French:
 - gamma_fair_abs_('Soya_beans_grain',_'French')
 + x_('Soya_beans_grain',_'French') <= 2.76923076923
FairAbs_plus_Soya_beans_grain_bovines:
 - gamma_fair_abs_('Soya_beans_grain',_'bovines')
 + x_('Soya_beans_grain',_'bovines') <= 11.0769230769
FairAbs_plus_Soya_beans_grain_porcines:
 - gamma_fair_abs_('Soya_beans_grain',_'porcines')
 + x_('Soya_beans_grain',_'porcines') <= 22.1538461538
FairAbs_plus_Wheat_grain_French: - gamma_fair_abs_('Wheat_grain',_'French')
 + x_('Wheat_grain',_'French') <= 195
FairAbs_plus_Wheat_grain_bovines: - gamma_fair_abs_('Wheat_grain',_'bovines')
 + x_('Wheat_grain',_'bovines') <= 97.5
FairAbs_plus_Wheat_grain_porcines:
 - gamma_fair_abs_('Wheat_grain',_'porcines') + x_('Wheat_grain',_'porcines')
 <= 195
NoSwap_I_Barley_grain: I_('Barley_grain',_'French')
 + I_('Barley_grain',_'bovines') + I_('Barley_grain',_'porcines')
 - 4176.07257236 y_Barley_grain <= 0
NoSwap_I_Forage: I_('Forage',_'bovines') + I_energy_Forage_Bioraffinery
 - 3159.0880698 y_Forage <= 0
NoSwap_I_Maize_corn: I_('Maize_corn',_'French') + I_('Maize_corn',_'porcines')
 - 1107.24091382 y_Maize_corn <= 0
NoSwap_I_Meat_of_cattle_with_the_bone_fresh_or_chilled:
 I_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 - 200.000001 y_Meat_of_cattle_with_the_bone_fresh_or_chilled <= 0
NoSwap_I_Meat_of_pig_with_the_bone_fresh_or_chilled:
 I_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 - 200.000001 y_Meat_of_pig_with_the_bone_fresh_or_chilled <= 0
NoSwap_I_Natural_meadow_forage: I_('Natural_meadow_forage',_'bovines')
 - 3141.0880698 y_Natural_meadow_forage <= 0
NoSwap_I_Non_edible_porcines_carcass: - 1e-06 y_Non_edible_porcines_carcass
 <= 0
NoSwap_I_Oats_grain: I_('Oats_grain',_'French') - 226.666667667 y_Oats_grain
 <= 0
NoSwap_I_Raw_milk_of_cattle: I_('Raw_milk_of_cattle',_'French')
 - 200.000001 y_Raw_milk_of_cattle <= 0
NoSwap_I_Rice_grain: I_('Rice_grain',_'French') - 200.000001 y_Rice_grain <= 0
NoSwap_I_Soya_beans_grain: I_('Soya_beans_grain',_'French')
 + I_('Soya_beans_grain',_'bovines') + I_('Soya_beans_grain',_'porcines')
 - 4153.5084698 y_Soya_beans_grain <= 0
NoSwap_I_Wheat_grain: I_('Wheat_grain',_'French')
 + I_('Wheat_grain',_'bovines') + I_('Wheat_grain',_'porcines')
 + I_energy_Wheat_grain_Bioraffinery - 4198.0084698 y_Wheat_grain <= 0
NoSwap_U_Barley_grain: U_Barley_grain + 250.8 y_Barley_grain <= 250.8
NoSwap_U_Forage: U_Forage <= 0
NoSwap_U_Maize_corn: U_Maize_corn + 71.25 y_Maize_corn <= 71.25
NoSwap_U_Meat_of_cattle_with_the_bone_fresh_or_chilled:
 U_Meat_of_cattle_with_the_bone_fresh_or_chilled
 + 31.22612 y_Meat_of_cattle_with_the_bone_fresh_or_chilled <= 31.22612
NoSwap_U_Meat_of_pig_with_the_bone_fresh_or_chilled:
 U_Meat_of_pig_with_the_bone_fresh_or_chilled
 + 31.93938 y_Meat_of_pig_with_the_bone_fresh_or_chilled <= 31.93938
NoSwap_U_Natural_meadow_forage: U_Natural_meadow_forage
 + 190.65 y_Natural_meadow_forage <= 190.65
NoSwap_U_Non_edible_porcines_carcass: U_Non_edible_porcines_carcass
 + 0.475 y_Non_edible_porcines_carcass <= 0.475
NoSwap_U_Oats_grain: U_Oats_grain + 51.1920344 y_Oats_grain <= 51.1920344
NoSwap_U_Raw_milk_of_cattle: U_Raw_milk_of_cattle
 + 96.01390536 y_Raw_milk_of_cattle <= 96.01390536
NoSwap_U_Rice_grain: U_Rice_grain + 1.35895885 y_Rice_grain <= 1.35895885
NoSwap_U_Soya_beans_grain: U_Soya_beans_grain + 34.2 y_Soya_beans_grain
 <= 34.2
NoSwap_U_Wheat_grain: U_Wheat_grain + 463.125 y_Wheat_grain <= 463.125
Pas_d_import_prairies_nat: I_('Natural_meadow_forage',_'bovines') = 0
Penalite_Culture_Moins_French_0.025_Barley_grain:
 - I_('Barley_grain',_'French')
 - 5 penalite_culture_('French',_0.025,_'Barley_grain')
 - x_('Barley_grain',_'French') <= -5
Penalite_Culture_Moins_French_0.025_Oats_grain: - I_('Oats_grain',_'French')
 - 5 penalite_culture_('French',_0.025,_'Oats_grain')
 - x_('Oats_grain',_'French') <= -5
Penalite_Culture_Moins_French_0.05_Maize_corn: - I_('Maize_corn',_'French')
 - 10 penalite_culture_('French',_0.05,_'Maize_corn')
 - x_('Maize_corn',_'French') <= -10
Penalite_Culture_Moins_French_0.05_Raw_milk_of_cattle:
 - I_('Raw_milk_of_cattle',_'French')
 - 10 penalite_culture_('French',_0.05,_'Raw_milk_of_cattle')
 - x_('Raw_milk_of_cattle',_'French') <= -10
Penalite_Culture_Moins_French_0.05_Rice_grain: - I_('Rice_grain',_'French')
 - 10 penalite_culture_('French',_0.05,_'Rice_grain')
 - x_('Rice_grain',_'French') <= -10
Penalite_Culture_Moins_French_0.05_Soya_beans_grain:
 - I_('Soya_beans_grain',_'French')
 - 10 penalite_culture_('French',_0.05,_'Soya_beans_grain')
 - x_('Soya_beans_grain',_'French') <= -10
Penalite_Culture_Moins_French_0.15_Meat_of_pig_with_the_bone_fresh_or_chilled:
 - I_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 - 30 penalite_culture_('French',_0.15,_'Meat_of_pig_with_the_bone_fresh_or_chilled')
 - x_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French') <= -30
Penalite_Culture_Moins_French_0.3_Meat_of_cattle_with_the_bone_fresh_or_chilled:
 - I_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 - 60 penalite_culture_('French',_0.3,_'Meat_of_cattle_with_the_bone_fresh_or_chilled')
 - x_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French') <= -60
Penalite_Culture_Moins_French_0.3_Wheat_grain: - I_('Wheat_grain',_'French')
 - 60 penalite_culture_('French',_0.3,_'Wheat_grain')
 - x_('Wheat_grain',_'French') <= -60
Penalite_Culture_Moins_bovines_0.2_Soya_beans_grain:
 - I_('Soya_beans_grain',_'bovines')
 - 611.81761376 penalite_culture_('bovines',_0.2,_'Soya_beans_grain')
 - x_('Soya_beans_grain',_'bovines') <= -611.81761376
Penalite_Culture_Moins_bovines_0.3_Barley_grain:
 - I_('Barley_grain',_'bovines')
 - 458.86321032 penalite_culture_('bovines',_0.3,_'Barley_grain')
 - x_('Barley_grain',_'bovines') <= -458.86321032
Penalite_Culture_Moins_bovines_0.3_Wheat_grain: - I_('Wheat_grain',_'bovines')
 - 458.86321032 penalite_culture_('bovines',_0.3,_'Wheat_grain')
 - x_('Wheat_grain',_'bovines') <= -458.86321032
Penalite_Culture_Moins_bovines_0.5_Forage: - I_('Forage',_'bovines')
 - 764.7720172 penalite_culture_('bovines',_0.5,_'Forage')
 - x_('Forage',_'bovines') <= -764.7720172
Penalite_Culture_Moins_bovines_0.5_Natural_meadow_forage:
 - I_('Natural_meadow_forage',_'bovines')
 - 764.7720172 penalite_culture_('bovines',_0.5,_'Natural_meadow_forage')
 - x_('Natural_meadow_forage',_'bovines') <= -764.7720172
Penalite_Culture_Moins_porcines_0.1_Maize_corn: - I_('Maize_corn',_'porcines')
 - 89.44204 penalite_culture_('porcines',_0.1,_'Maize_corn')
 - x_('Maize_corn',_'porcines') <= -89.44204
Penalite_Culture_Moins_porcines_0.2_Barley_grain:
 - I_('Barley_grain',_'porcines')
 - 178.88408 penalite_culture_('porcines',_0.2,_'Barley_grain')
 - x_('Barley_grain',_'porcines') <= -178.88408
Penalite_Culture_Moins_porcines_0.3_Wheat_grain:
 - I_('Wheat_grain',_'porcines')
 - 268.32612 penalite_culture_('porcines',_0.3,_'Wheat_grain')
 - x_('Wheat_grain',_'porcines') <= -268.32612
Penalite_Culture_Moins_porcines_0.4_Soya_beans_grain:
 - I_('Soya_beans_grain',_'porcines')
 - 357.76816 penalite_culture_('porcines',_0.4,_'Soya_beans_grain')
 - x_('Soya_beans_grain',_'porcines') <= -357.76816
Penalite_Culture_Plus_French_0.025_Barley_grain: I_('Barley_grain',_'French')
 - 5 penalite_culture_('French',_0.025,_'Barley_grain')
 + x_('Barley_grain',_'French') <= 5
Penalite_Culture_Plus_French_0.025_Oats_grain: I_('Oats_grain',_'French')
 - 5 penalite_culture_('French',_0.025,_'Oats_grain')
 + x_('Oats_grain',_'French') <= 5
Penalite_Culture_Plus_French_0.05_Maize_corn: I_('Maize_corn',_'French')
 - 10 penalite_culture_('French',_0.05,_'Maize_corn')
 + x_('Maize_corn',_'French') <= 10
Penalite_Culture_Plus_French_0.05_Raw_milk_of_cattle:
 I_('Raw_milk_of_cattle',_'French')
 - 10 penalite_culture_('French',_0.05,_'Raw_milk_of_cattle')
 + x_('Raw_milk_of_cattle',_'French') <= 10
Penalite_Culture_Plus_French_0.05_Rice_grain: I_('Rice_grain',_'French')
 - 10 penalite_culture_('French',_0.05,_'Rice_grain')
 + x_('Rice_grain',_'French') <= 10
Penalite_Culture_Plus_French_0.05_Soya_beans_grain:
 I_('Soya_beans_grain',_'French')
 - 10 penalite_culture_('French',_0.05,_'Soya_beans_grain')
 + x_('Soya_beans_grain',_'French') <= 10
Penalite_Culture_Plus_French_0.15_Meat_of_pig_with_the_bone_fresh_or_chilled:
 I_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French')
 - 30 penalite_culture_('French',_0.15,_'Meat_of_pig_with_the_bone_fresh_or_chilled')
 + x_('Meat_of_pig_with_the_bone_fresh_or_chilled',_'French') <= 30
Penalite_Culture_Plus_French_0.3_Meat_of_cattle_with_the_bone_fresh_or_chilled:
 I_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French')
 - 60 penalite_culture_('French',_0.3,_'Meat_of_cattle_with_the_bone_fresh_or_chilled')
 + x_('Meat_of_cattle_with_the_bone_fresh_or_chilled',_'French') <= 60
Penalite_Culture_Plus_French_0.3_Wheat_grain: I_('Wheat_grain',_'French')
 - 60 penalite_culture_('French',_0.3,_'Wheat_grain')
 + x_('Wheat_grain',_'French') <= 60
Penalite_Culture_Plus_bovines_0.2_Soya_beans_grain:
 I_('Soya_beans_grain',_'bovines')
 - 611.81761376 penalite_culture_('bovines',_0.2,_'Soya_beans_grain')
 + x_('Soya_beans_grain',_'bovines') <= 611.81761376
Penalite_Culture_Plus_bovines_0.3_Barley_grain: I_('Barley_grain',_'bovines')
 - 458.86321032 penalite_culture_('bovines',_0.3,_'Barley_grain')
 + x_('Barley_grain',_'bovines') <= 458.86321032
Penalite_Culture_Plus_bovines_0.3_Wheat_grain: I_('Wheat_grain',_'bovines')
 - 458.86321032 penalite_culture_('bovines',_0.3,_'Wheat_grain')
 + x_('Wheat_grain',_'bovines') <= 458.86321032
Penalite_Culture_Plus_bovines_0.5_Forage: I_('Forage',_'bovines')
 - 764.7720172 penalite_culture_('bovines',_0.5,_'Forage')
 + x_('Forage',_'bovines') <= 764.7720172
Penalite_Culture_Plus_bovines_0.5_Natural_meadow_forage:
 I_('Natural_meadow_forage',_'bovines')
 - 764.7720172 penalite_culture_('bovines',_0.5,_'Natural_meadow_forage')
 + x_('Natural_meadow_forage',_'bovines') <= 764.7720172
Penalite_Culture_Plus_porcines_0.1_Maize_corn: I_('Maize_corn',_'porcines')
 - 89.44204 penalite_culture_('porcines',_0.1,_'Maize_corn')
 + x_('Maize_corn',_'porcines') <= 89.44204
Penalite_Culture_Plus_porcines_0.2_Barley_grain:
 I_('Barley_grain',_'porcines')
 - 178.88408 penalite_culture_('porcines',_0.2,_'Barley_grain')
 + x_('Barley_grain',_'porcines') <= 178.88408
Penalite_Culture_Plus_porcines_0.3_Wheat_grain: I_('Wheat_grain',_'porcines')
 - 268.32612 penalite_culture_('porcines',_0.3,_'Wheat_grain')
 + x_('Wheat_grain',_'porcines') <= 268.32612
Penalite_Culture_Plus_porcines_0.4_Soya_beans_grain:
 I_('Soya_beans_grain',_'porcines')
 - 357.76816 penalite_culture_('porcines',_0.4,_'Soya_beans_grain')
 + x_('Soya_beans_grain',_'porcines') <= 357.76816
Penalite_Energy_Moins_Bioraffinery__444243779325319801:
 0.5 I_energy_Forage_Bioraffinery + 0.5 I_energy_Wheat_grain_Bioraffinery
 - penalite_culture_Bioraffinery_('Bioraffinery',_0.5,_'Wheat_grain')
 + 0.5 x_Bioraffinery_prod_Forage - 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Bioraffinery__8649370978988607227:
 0.5 I_energy_Forage_Bioraffinery + 0.5 I_energy_Wheat_grain_Bioraffinery
 - penalite_culture_Bioraffinery_('Bioraffinery',_0.5,_'Forage')
 - 0.5 x_Bioraffinery_prod_Forage + 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Methanizer_1181331534250931803: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Oats_grain')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 + 0.05 x_Methanizer_prod_Barley_grain + 0.05 x_Methanizer_prod_Forage
 + 0.05 x_Methanizer_prod_Maize_corn
 + 0.05 x_Methanizer_prod_Natural_meadow_forage
 - 0.95 x_Methanizer_prod_Oats_grain + 0.05 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Methanizer_2935839473461582677: 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'porcines_slurry')
 + 0.15 x_Methanizer_excr_bovines_manure
 + 0.15 x_Methanizer_excr_bovines_slurry
 + 0.15 x_Methanizer_excr_porcines_manure
 - 0.85 x_Methanizer_excr_porcines_slurry
 + 0.15 x_Methanizer_prod_Barley_grain + 0.15 x_Methanizer_prod_Forage
 + 0.15 x_Methanizer_prod_Maize_corn
 + 0.15 x_Methanizer_prod_Natural_meadow_forage
 + 0.15 x_Methanizer_prod_Oats_grain + 0.15 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Methanizer_299828619908280580: 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'bovines_slurry')
 + 0.15 x_Methanizer_excr_bovines_manure
 - 0.85 x_Methanizer_excr_bovines_slurry
 + 0.15 x_Methanizer_excr_porcines_manure
 + 0.15 x_Methanizer_excr_porcines_slurry
 + 0.15 x_Methanizer_prod_Barley_grain + 0.15 x_Methanizer_prod_Forage
 + 0.15 x_Methanizer_prod_Maize_corn
 + 0.15 x_Methanizer_prod_Natural_meadow_forage
 + 0.15 x_Methanizer_prod_Oats_grain + 0.15 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Methanizer_5556847830181414960: 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'bovines_manure')
 - 0.85 x_Methanizer_excr_bovines_manure
 + 0.15 x_Methanizer_excr_bovines_slurry
 + 0.15 x_Methanizer_excr_porcines_manure
 + 0.15 x_Methanizer_excr_porcines_slurry
 + 0.15 x_Methanizer_prod_Barley_grain + 0.15 x_Methanizer_prod_Forage
 + 0.15 x_Methanizer_prod_Maize_corn
 + 0.15 x_Methanizer_prod_Natural_meadow_forage
 + 0.15 x_Methanizer_prod_Oats_grain + 0.15 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Methanizer_7567851437475611983: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Maize_corn')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 + 0.05 x_Methanizer_prod_Barley_grain + 0.05 x_Methanizer_prod_Forage
 - 0.95 x_Methanizer_prod_Maize_corn
 + 0.05 x_Methanizer_prod_Natural_meadow_forage
 + 0.05 x_Methanizer_prod_Oats_grain + 0.05 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Methanizer_7629943078603667174: - 0.9 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'waste')
 + 0.1 x_Methanizer_excr_bovines_manure + 0.1 x_Methanizer_excr_bovines_slurry
 + 0.1 x_Methanizer_excr_porcines_manure
 + 0.1 x_Methanizer_excr_porcines_slurry + 0.1 x_Methanizer_prod_Barley_grain
 + 0.1 x_Methanizer_prod_Forage + 0.1 x_Methanizer_prod_Maize_corn
 + 0.1 x_Methanizer_prod_Natural_meadow_forage
 + 0.1 x_Methanizer_prod_Oats_grain + 0.1 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Methanizer_776380305261090420: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Barley_grain')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 - 0.95 x_Methanizer_prod_Barley_grain + 0.05 x_Methanizer_prod_Forage
 + 0.05 x_Methanizer_prod_Maize_corn
 + 0.05 x_Methanizer_prod_Natural_meadow_forage
 + 0.05 x_Methanizer_prod_Oats_grain + 0.05 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Methanizer_7928357584263881846: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Wheat_grain')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 + 0.05 x_Methanizer_prod_Barley_grain + 0.05 x_Methanizer_prod_Forage
 + 0.05 x_Methanizer_prod_Maize_corn
 + 0.05 x_Methanizer_prod_Natural_meadow_forage
 + 0.05 x_Methanizer_prod_Oats_grain - 0.95 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Methanizer__2072015263469454467: 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'porcines_manure')
 + 0.15 x_Methanizer_excr_bovines_manure
 + 0.15 x_Methanizer_excr_bovines_slurry
 - 0.85 x_Methanizer_excr_porcines_manure
 + 0.15 x_Methanizer_excr_porcines_slurry
 + 0.15 x_Methanizer_prod_Barley_grain + 0.15 x_Methanizer_prod_Forage
 + 0.15 x_Methanizer_prod_Maize_corn
 + 0.15 x_Methanizer_prod_Natural_meadow_forage
 + 0.15 x_Methanizer_prod_Oats_grain + 0.15 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Methanizer__5876890340316322018: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'Natural_meadow_forage')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 + 0.05 x_Methanizer_prod_Barley_grain + 0.05 x_Methanizer_prod_Forage
 + 0.05 x_Methanizer_prod_Maize_corn
 - 0.95 x_Methanizer_prod_Natural_meadow_forage
 + 0.05 x_Methanizer_prod_Oats_grain + 0.05 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Moins_Methanizer__6852447449822079796: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'Forage')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 + 0.05 x_Methanizer_prod_Barley_grain - 0.95 x_Methanizer_prod_Forage
 + 0.05 x_Methanizer_prod_Maize_corn
 + 0.05 x_Methanizer_prod_Natural_meadow_forage
 + 0.05 x_Methanizer_prod_Oats_grain + 0.05 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Bioraffinery__444243779325319801:
 - 0.5 I_energy_Forage_Bioraffinery - 0.5 I_energy_Wheat_grain_Bioraffinery
 - penalite_culture_Bioraffinery_('Bioraffinery',_0.5,_'Wheat_grain')
 - 0.5 x_Bioraffinery_prod_Forage + 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Bioraffinery__8649370978988607227:
 - 0.5 I_energy_Forage_Bioraffinery - 0.5 I_energy_Wheat_grain_Bioraffinery
 - penalite_culture_Bioraffinery_('Bioraffinery',_0.5,_'Forage')
 + 0.5 x_Bioraffinery_prod_Forage - 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Methanizer_1181331534250931803: - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Oats_grain')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 - 0.05 x_Methanizer_prod_Barley_grain - 0.05 x_Methanizer_prod_Forage
 - 0.05 x_Methanizer_prod_Maize_corn
 - 0.05 x_Methanizer_prod_Natural_meadow_forage
 + 0.95 x_Methanizer_prod_Oats_grain - 0.05 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Methanizer_2935839473461582677: - 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'porcines_slurry')
 - 0.15 x_Methanizer_excr_bovines_manure
 - 0.15 x_Methanizer_excr_bovines_slurry
 - 0.15 x_Methanizer_excr_porcines_manure
 + 0.85 x_Methanizer_excr_porcines_slurry
 - 0.15 x_Methanizer_prod_Barley_grain - 0.15 x_Methanizer_prod_Forage
 - 0.15 x_Methanizer_prod_Maize_corn
 - 0.15 x_Methanizer_prod_Natural_meadow_forage
 - 0.15 x_Methanizer_prod_Oats_grain - 0.15 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Methanizer_299828619908280580: - 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'bovines_slurry')
 - 0.15 x_Methanizer_excr_bovines_manure
 + 0.85 x_Methanizer_excr_bovines_slurry
 - 0.15 x_Methanizer_excr_porcines_manure
 - 0.15 x_Methanizer_excr_porcines_slurry
 - 0.15 x_Methanizer_prod_Barley_grain - 0.15 x_Methanizer_prod_Forage
 - 0.15 x_Methanizer_prod_Maize_corn
 - 0.15 x_Methanizer_prod_Natural_meadow_forage
 - 0.15 x_Methanizer_prod_Oats_grain - 0.15 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Methanizer_5556847830181414960: - 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'bovines_manure')
 + 0.85 x_Methanizer_excr_bovines_manure
 - 0.15 x_Methanizer_excr_bovines_slurry
 - 0.15 x_Methanizer_excr_porcines_manure
 - 0.15 x_Methanizer_excr_porcines_slurry
 - 0.15 x_Methanizer_prod_Barley_grain - 0.15 x_Methanizer_prod_Forage
 - 0.15 x_Methanizer_prod_Maize_corn
 - 0.15 x_Methanizer_prod_Natural_meadow_forage
 - 0.15 x_Methanizer_prod_Oats_grain - 0.15 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Methanizer_7567851437475611983: - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Maize_corn')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 - 0.05 x_Methanizer_prod_Barley_grain - 0.05 x_Methanizer_prod_Forage
 + 0.95 x_Methanizer_prod_Maize_corn
 - 0.05 x_Methanizer_prod_Natural_meadow_forage
 - 0.05 x_Methanizer_prod_Oats_grain - 0.05 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Methanizer_7629943078603667174: 0.9 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'waste')
 - 0.1 x_Methanizer_excr_bovines_manure - 0.1 x_Methanizer_excr_bovines_slurry
 - 0.1 x_Methanizer_excr_porcines_manure
 - 0.1 x_Methanizer_excr_porcines_slurry - 0.1 x_Methanizer_prod_Barley_grain
 - 0.1 x_Methanizer_prod_Forage - 0.1 x_Methanizer_prod_Maize_corn
 - 0.1 x_Methanizer_prod_Natural_meadow_forage
 - 0.1 x_Methanizer_prod_Oats_grain - 0.1 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Methanizer_776380305261090420: - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Barley_grain')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 + 0.95 x_Methanizer_prod_Barley_grain - 0.05 x_Methanizer_prod_Forage
 - 0.05 x_Methanizer_prod_Maize_corn
 - 0.05 x_Methanizer_prod_Natural_meadow_forage
 - 0.05 x_Methanizer_prod_Oats_grain - 0.05 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Methanizer_7928357584263881846: - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Wheat_grain')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 - 0.05 x_Methanizer_prod_Barley_grain - 0.05 x_Methanizer_prod_Forage
 - 0.05 x_Methanizer_prod_Maize_corn
 - 0.05 x_Methanizer_prod_Natural_meadow_forage
 - 0.05 x_Methanizer_prod_Oats_grain + 0.95 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Methanizer__2072015263469454467:
 - 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'porcines_manure')
 - 0.15 x_Methanizer_excr_bovines_manure
 - 0.15 x_Methanizer_excr_bovines_slurry
 + 0.85 x_Methanizer_excr_porcines_manure
 - 0.15 x_Methanizer_excr_porcines_slurry
 - 0.15 x_Methanizer_prod_Barley_grain - 0.15 x_Methanizer_prod_Forage
 - 0.15 x_Methanizer_prod_Maize_corn
 - 0.15 x_Methanizer_prod_Natural_meadow_forage
 - 0.15 x_Methanizer_prod_Oats_grain - 0.15 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Methanizer__5876890340316322018:
 - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'Natural_meadow_forage')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 - 0.05 x_Methanizer_prod_Barley_grain - 0.05 x_Methanizer_prod_Forage
 - 0.05 x_Methanizer_prod_Maize_corn
 + 0.95 x_Methanizer_prod_Natural_meadow_forage
 - 0.05 x_Methanizer_prod_Oats_grain - 0.05 x_Methanizer_prod_Wheat_grain <= 0
Penalite_Energy_Plus_Methanizer__6852447449822079796:
 - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'Forage')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 - 0.05 x_Methanizer_prod_Barley_grain + 0.95 x_Methanizer_prod_Forage
 - 0.05 x_Methanizer_prod_Maize_corn
 - 0.05 x_Methanizer_prod_Natural_meadow_forage
 - 0.05 x_Methanizer_prod_Oats_grain - 0.05 x_Methanizer_prod_Wheat_grain <= 0
_C1: Methanizer_energy_dev - 0.1 N_Methanizer_waste
 - 0.046 x_Methanizer_excr_bovines_manure
 - 0.0766666666667 x_Methanizer_excr_bovines_slurry
 - 0.02875 x_Methanizer_excr_porcines_manure
 - 0.046 x_Methanizer_excr_porcines_slurry
 - 0.0443181818182 x_Methanizer_prod_Barley_grain
 - 0.0125 x_Methanizer_prod_Forage - 0.078 x_Methanizer_prod_Maize_corn
 - 0.0121951219512 x_Methanizer_prod_Natural_meadow_forage
 - 0.0375 x_Methanizer_prod_Oats_grain - 0.04 x_Methanizer_prod_Wheat_grain
 >= -1
_C10: 0.1 N_Methanizer_waste - delta_Methanizer_3
 + 0.1 x_Methanizer_excr_bovines_manure + 0.1 x_Methanizer_excr_bovines_slurry
 + 0.1 x_Methanizer_excr_porcines_manure
 + 0.1 x_Methanizer_excr_porcines_slurry + 0.1 x_Methanizer_prod_Barley_grain
 - 0.9 x_Methanizer_prod_Forage + 0.1 x_Methanizer_prod_Maize_corn
 - 0.9 x_Methanizer_prod_Natural_meadow_forage
 + 0.1 x_Methanizer_prod_Oats_grain + 0.1 x_Methanizer_prod_Wheat_grain <= 0
_C11: - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Barley_grain')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 + 0.95 x_Methanizer_prod_Barley_grain - 0.05 x_Methanizer_prod_Forage
 - 0.05 x_Methanizer_prod_Maize_corn
 - 0.05 x_Methanizer_prod_Natural_meadow_forage
 - 0.05 x_Methanizer_prod_Oats_grain - 0.05 x_Methanizer_prod_Wheat_grain <= 0
_C12: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Barley_grain')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 - 0.95 x_Methanizer_prod_Barley_grain + 0.05 x_Methanizer_prod_Forage
 + 0.05 x_Methanizer_prod_Maize_corn
 + 0.05 x_Methanizer_prod_Natural_meadow_forage
 + 0.05 x_Methanizer_prod_Oats_grain + 0.05 x_Methanizer_prod_Wheat_grain <= 0
_C13: - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Wheat_grain')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 - 0.05 x_Methanizer_prod_Barley_grain - 0.05 x_Methanizer_prod_Forage
 - 0.05 x_Methanizer_prod_Maize_corn
 - 0.05 x_Methanizer_prod_Natural_meadow_forage
 - 0.05 x_Methanizer_prod_Oats_grain + 0.95 x_Methanizer_prod_Wheat_grain <= 0
_C14: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Wheat_grain')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 + 0.05 x_Methanizer_prod_Barley_grain + 0.05 x_Methanizer_prod_Forage
 + 0.05 x_Methanizer_prod_Maize_corn
 + 0.05 x_Methanizer_prod_Natural_meadow_forage
 + 0.05 x_Methanizer_prod_Oats_grain - 0.95 x_Methanizer_prod_Wheat_grain <= 0
_C15: - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Oats_grain')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 - 0.05 x_Methanizer_prod_Barley_grain - 0.05 x_Methanizer_prod_Forage
 - 0.05 x_Methanizer_prod_Maize_corn
 - 0.05 x_Methanizer_prod_Natural_meadow_forage
 + 0.95 x_Methanizer_prod_Oats_grain - 0.05 x_Methanizer_prod_Wheat_grain <= 0
_C16: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Oats_grain')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 + 0.05 x_Methanizer_prod_Barley_grain + 0.05 x_Methanizer_prod_Forage
 + 0.05 x_Methanizer_prod_Maize_corn
 + 0.05 x_Methanizer_prod_Natural_meadow_forage
 - 0.95 x_Methanizer_prod_Oats_grain + 0.05 x_Methanizer_prod_Wheat_grain <= 0
_C17: - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Maize_corn')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 - 0.05 x_Methanizer_prod_Barley_grain - 0.05 x_Methanizer_prod_Forage
 + 0.95 x_Methanizer_prod_Maize_corn
 - 0.05 x_Methanizer_prod_Natural_meadow_forage
 - 0.05 x_Methanizer_prod_Oats_grain - 0.05 x_Methanizer_prod_Wheat_grain <= 0
_C18: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.2,_'Maize_corn')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 + 0.05 x_Methanizer_prod_Barley_grain + 0.05 x_Methanizer_prod_Forage
 - 0.95 x_Methanizer_prod_Maize_corn
 + 0.05 x_Methanizer_prod_Natural_meadow_forage
 + 0.05 x_Methanizer_prod_Oats_grain + 0.05 x_Methanizer_prod_Wheat_grain <= 0
_C19: 0.9 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'waste')
 - 0.1 x_Methanizer_excr_bovines_manure - 0.1 x_Methanizer_excr_bovines_slurry
 - 0.1 x_Methanizer_excr_porcines_manure
 - 0.1 x_Methanizer_excr_porcines_slurry - 0.1 x_Methanizer_prod_Barley_grain
 - 0.1 x_Methanizer_prod_Forage - 0.1 x_Methanizer_prod_Maize_corn
 - 0.1 x_Methanizer_prod_Natural_meadow_forage
 - 0.1 x_Methanizer_prod_Oats_grain - 0.1 x_Methanizer_prod_Wheat_grain <= 0
_C2: Methanizer_energy_dev + 0.1 N_Methanizer_waste
 + 0.046 x_Methanizer_excr_bovines_manure
 + 0.0766666666667 x_Methanizer_excr_bovines_slurry
 + 0.02875 x_Methanizer_excr_porcines_manure
 + 0.046 x_Methanizer_excr_porcines_slurry
 + 0.0443181818182 x_Methanizer_prod_Barley_grain
 + 0.0125 x_Methanizer_prod_Forage + 0.078 x_Methanizer_prod_Maize_corn
 + 0.0121951219512 x_Methanizer_prod_Natural_meadow_forage
 + 0.0375 x_Methanizer_prod_Oats_grain + 0.04 x_Methanizer_prod_Wheat_grain
 >= 1
_C20: - 0.9 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'waste')
 + 0.1 x_Methanizer_excr_bovines_manure + 0.1 x_Methanizer_excr_bovines_slurry
 + 0.1 x_Methanizer_excr_porcines_manure
 + 0.1 x_Methanizer_excr_porcines_slurry + 0.1 x_Methanizer_prod_Barley_grain
 + 0.1 x_Methanizer_prod_Forage + 0.1 x_Methanizer_prod_Maize_corn
 + 0.1 x_Methanizer_prod_Natural_meadow_forage
 + 0.1 x_Methanizer_prod_Oats_grain + 0.1 x_Methanizer_prod_Wheat_grain <= 0
_C21: - 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'bovines_slurry')
 - 0.15 x_Methanizer_excr_bovines_manure
 + 0.85 x_Methanizer_excr_bovines_slurry
 - 0.15 x_Methanizer_excr_porcines_manure
 - 0.15 x_Methanizer_excr_porcines_slurry
 - 0.15 x_Methanizer_prod_Barley_grain - 0.15 x_Methanizer_prod_Forage
 - 0.15 x_Methanizer_prod_Maize_corn
 - 0.15 x_Methanizer_prod_Natural_meadow_forage
 - 0.15 x_Methanizer_prod_Oats_grain - 0.15 x_Methanizer_prod_Wheat_grain <= 0
_C22: 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'bovines_slurry')
 + 0.15 x_Methanizer_excr_bovines_manure
 - 0.85 x_Methanizer_excr_bovines_slurry
 + 0.15 x_Methanizer_excr_porcines_manure
 + 0.15 x_Methanizer_excr_porcines_slurry
 + 0.15 x_Methanizer_prod_Barley_grain + 0.15 x_Methanizer_prod_Forage
 + 0.15 x_Methanizer_prod_Maize_corn
 + 0.15 x_Methanizer_prod_Natural_meadow_forage
 + 0.15 x_Methanizer_prod_Oats_grain + 0.15 x_Methanizer_prod_Wheat_grain <= 0
_C23: - 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'bovines_manure')
 + 0.85 x_Methanizer_excr_bovines_manure
 - 0.15 x_Methanizer_excr_bovines_slurry
 - 0.15 x_Methanizer_excr_porcines_manure
 - 0.15 x_Methanizer_excr_porcines_slurry
 - 0.15 x_Methanizer_prod_Barley_grain - 0.15 x_Methanizer_prod_Forage
 - 0.15 x_Methanizer_prod_Maize_corn
 - 0.15 x_Methanizer_prod_Natural_meadow_forage
 - 0.15 x_Methanizer_prod_Oats_grain - 0.15 x_Methanizer_prod_Wheat_grain <= 0
_C24: 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'bovines_manure')
 - 0.85 x_Methanizer_excr_bovines_manure
 + 0.15 x_Methanizer_excr_bovines_slurry
 + 0.15 x_Methanizer_excr_porcines_manure
 + 0.15 x_Methanizer_excr_porcines_slurry
 + 0.15 x_Methanizer_prod_Barley_grain + 0.15 x_Methanizer_prod_Forage
 + 0.15 x_Methanizer_prod_Maize_corn
 + 0.15 x_Methanizer_prod_Natural_meadow_forage
 + 0.15 x_Methanizer_prod_Oats_grain + 0.15 x_Methanizer_prod_Wheat_grain <= 0
_C25: - 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'porcines_slurry')
 - 0.15 x_Methanizer_excr_bovines_manure
 - 0.15 x_Methanizer_excr_bovines_slurry
 - 0.15 x_Methanizer_excr_porcines_manure
 + 0.85 x_Methanizer_excr_porcines_slurry
 - 0.15 x_Methanizer_prod_Barley_grain - 0.15 x_Methanizer_prod_Forage
 - 0.15 x_Methanizer_prod_Maize_corn
 - 0.15 x_Methanizer_prod_Natural_meadow_forage
 - 0.15 x_Methanizer_prod_Oats_grain - 0.15 x_Methanizer_prod_Wheat_grain <= 0
_C26: 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'porcines_slurry')
 + 0.15 x_Methanizer_excr_bovines_manure
 + 0.15 x_Methanizer_excr_bovines_slurry
 + 0.15 x_Methanizer_excr_porcines_manure
 - 0.85 x_Methanizer_excr_porcines_slurry
 + 0.15 x_Methanizer_prod_Barley_grain + 0.15 x_Methanizer_prod_Forage
 + 0.15 x_Methanizer_prod_Maize_corn
 + 0.15 x_Methanizer_prod_Natural_meadow_forage
 + 0.15 x_Methanizer_prod_Oats_grain + 0.15 x_Methanizer_prod_Wheat_grain <= 0
_C27: - 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'porcines_manure')
 - 0.15 x_Methanizer_excr_bovines_manure
 - 0.15 x_Methanizer_excr_bovines_slurry
 + 0.85 x_Methanizer_excr_porcines_manure
 - 0.15 x_Methanizer_excr_porcines_slurry
 - 0.15 x_Methanizer_prod_Barley_grain - 0.15 x_Methanizer_prod_Forage
 - 0.15 x_Methanizer_prod_Maize_corn
 - 0.15 x_Methanizer_prod_Natural_meadow_forage
 - 0.15 x_Methanizer_prod_Oats_grain - 0.15 x_Methanizer_prod_Wheat_grain <= 0
_C28: 0.15 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.6,_'porcines_manure')
 + 0.15 x_Methanizer_excr_bovines_manure
 + 0.15 x_Methanizer_excr_bovines_slurry
 - 0.85 x_Methanizer_excr_porcines_manure
 + 0.15 x_Methanizer_excr_porcines_slurry
 + 0.15 x_Methanizer_prod_Barley_grain + 0.15 x_Methanizer_prod_Forage
 + 0.15 x_Methanizer_prod_Maize_corn
 + 0.15 x_Methanizer_prod_Natural_meadow_forage
 + 0.15 x_Methanizer_prod_Oats_grain + 0.15 x_Methanizer_prod_Wheat_grain <= 0
_C29: - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'Natural_meadow_forage')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 - 0.05 x_Methanizer_prod_Barley_grain - 0.05 x_Methanizer_prod_Forage
 - 0.05 x_Methanizer_prod_Maize_corn
 + 0.95 x_Methanizer_prod_Natural_meadow_forage
 - 0.05 x_Methanizer_prod_Oats_grain - 0.05 x_Methanizer_prod_Wheat_grain <= 0
_C3: - 0.2 N_Methanizer_waste - delta_Methanizer_0
 - 0.2 x_Methanizer_excr_bovines_manure - 0.2 x_Methanizer_excr_bovines_slurry
 - 0.2 x_Methanizer_excr_porcines_manure
 - 0.2 x_Methanizer_excr_porcines_slurry + 0.8 x_Methanizer_prod_Barley_grain
 - 0.2 x_Methanizer_prod_Forage + 0.8 x_Methanizer_prod_Maize_corn
 - 0.2 x_Methanizer_prod_Natural_meadow_forage
 + 0.8 x_Methanizer_prod_Oats_grain + 0.8 x_Methanizer_prod_Wheat_grain <= 0
_C30: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'Natural_meadow_forage')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 + 0.05 x_Methanizer_prod_Barley_grain + 0.05 x_Methanizer_prod_Forage
 + 0.05 x_Methanizer_prod_Maize_corn
 - 0.95 x_Methanizer_prod_Natural_meadow_forage
 + 0.05 x_Methanizer_prod_Oats_grain + 0.05 x_Methanizer_prod_Wheat_grain <= 0
_C31: - 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'Forage')
 - 0.05 x_Methanizer_excr_bovines_manure
 - 0.05 x_Methanizer_excr_bovines_slurry
 - 0.05 x_Methanizer_excr_porcines_manure
 - 0.05 x_Methanizer_excr_porcines_slurry
 - 0.05 x_Methanizer_prod_Barley_grain + 0.95 x_Methanizer_prod_Forage
 - 0.05 x_Methanizer_prod_Maize_corn
 - 0.05 x_Methanizer_prod_Natural_meadow_forage
 - 0.05 x_Methanizer_prod_Oats_grain - 0.05 x_Methanizer_prod_Wheat_grain <= 0
_C32: 0.05 N_Methanizer_waste
 - penalite_culture_Methanizer_('Methanizer',_0.1,_'Forage')
 + 0.05 x_Methanizer_excr_bovines_manure
 + 0.05 x_Methanizer_excr_bovines_slurry
 + 0.05 x_Methanizer_excr_porcines_manure
 + 0.05 x_Methanizer_excr_porcines_slurry
 + 0.05 x_Methanizer_prod_Barley_grain - 0.95 x_Methanizer_prod_Forage
 + 0.05 x_Methanizer_prod_Maize_corn
 + 0.05 x_Methanizer_prod_Natural_meadow_forage
 + 0.05 x_Methanizer_prod_Oats_grain + 0.05 x_Methanizer_prod_Wheat_grain <= 0
_C33: Bioraffinery_energy_dev - 0.05 I_energy_Forage_Bioraffinery
 - 0.0512820512821 I_energy_Wheat_grain_Bioraffinery
 - 0.05 x_Bioraffinery_prod_Forage
 - 0.0512820512821 x_Bioraffinery_prod_Wheat_grain >= -1
_C34: Bioraffinery_energy_dev + 0.05 I_energy_Forage_Bioraffinery
 + 0.0512820512821 I_energy_Wheat_grain_Bioraffinery
 + 0.05 x_Bioraffinery_prod_Forage
 + 0.0512820512821 x_Bioraffinery_prod_Wheat_grain >= 1
_C35: 0.5 I_energy_Forage_Bioraffinery - 0.5 I_energy_Wheat_grain_Bioraffinery
 - delta_Bioraffinery_0 + 0.5 x_Bioraffinery_prod_Forage
 - 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
_C36: - 0.5 I_energy_Forage_Bioraffinery
 + 0.5 I_energy_Wheat_grain_Bioraffinery - delta_Bioraffinery_0
 - 0.5 x_Bioraffinery_prod_Forage + 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
_C37: - 0.5 I_energy_Forage_Bioraffinery
 + 0.5 I_energy_Wheat_grain_Bioraffinery - delta_Bioraffinery_1
 - 0.5 x_Bioraffinery_prod_Forage + 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
_C38: 0.5 I_energy_Forage_Bioraffinery - 0.5 I_energy_Wheat_grain_Bioraffinery
 - delta_Bioraffinery_1 + 0.5 x_Bioraffinery_prod_Forage
 - 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
_C39: - 0.5 I_energy_Forage_Bioraffinery
 - 0.5 I_energy_Wheat_grain_Bioraffinery
 - penalite_culture_Bioraffinery_('Bioraffinery',_0.5,_'Forage')
 + 0.5 x_Bioraffinery_prod_Forage - 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
_C4: 0.2 N_Methanizer_waste - delta_Methanizer_0
 + 0.2 x_Methanizer_excr_bovines_manure + 0.2 x_Methanizer_excr_bovines_slurry
 + 0.2 x_Methanizer_excr_porcines_manure
 + 0.2 x_Methanizer_excr_porcines_slurry - 0.8 x_Methanizer_prod_Barley_grain
 + 0.2 x_Methanizer_prod_Forage - 0.8 x_Methanizer_prod_Maize_corn
 + 0.2 x_Methanizer_prod_Natural_meadow_forage
 - 0.8 x_Methanizer_prod_Oats_grain - 0.8 x_Methanizer_prod_Wheat_grain <= 0
_C40: 0.5 I_energy_Forage_Bioraffinery + 0.5 I_energy_Wheat_grain_Bioraffinery
 - penalite_culture_Bioraffinery_('Bioraffinery',_0.5,_'Forage')
 - 0.5 x_Bioraffinery_prod_Forage + 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
_C41: - 0.5 I_energy_Forage_Bioraffinery
 - 0.5 I_energy_Wheat_grain_Bioraffinery
 - penalite_culture_Bioraffinery_('Bioraffinery',_0.5,_'Wheat_grain')
 - 0.5 x_Bioraffinery_prod_Forage + 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
_C42: 0.5 I_energy_Forage_Bioraffinery + 0.5 I_energy_Wheat_grain_Bioraffinery
 - penalite_culture_Bioraffinery_('Bioraffinery',_0.5,_'Wheat_grain')
 + 0.5 x_Bioraffinery_prod_Forage - 0.5 x_Bioraffinery_prod_Wheat_grain <= 0
_C5: 0.9 N_Methanizer_waste - delta_Methanizer_1
 - 0.1 x_Methanizer_excr_bovines_manure - 0.1 x_Methanizer_excr_bovines_slurry
 - 0.1 x_Methanizer_excr_porcines_manure
 - 0.1 x_Methanizer_excr_porcines_slurry - 0.1 x_Methanizer_prod_Barley_grain
 - 0.1 x_Methanizer_prod_Forage - 0.1 x_Methanizer_prod_Maize_corn
 - 0.1 x_Methanizer_prod_Natural_meadow_forage
 - 0.1 x_Methanizer_prod_Oats_grain - 0.1 x_Methanizer_prod_Wheat_grain <= 0
_C6: - 0.9 N_Methanizer_waste - delta_Methanizer_1
 + 0.1 x_Methanizer_excr_bovines_manure + 0.1 x_Methanizer_excr_bovines_slurry
 + 0.1 x_Methanizer_excr_porcines_manure
 + 0.1 x_Methanizer_excr_porcines_slurry + 0.1 x_Methanizer_prod_Barley_grain
 + 0.1 x_Methanizer_prod_Forage + 0.1 x_Methanizer_prod_Maize_corn
 + 0.1 x_Methanizer_prod_Natural_meadow_forage
 + 0.1 x_Methanizer_prod_Oats_grain + 0.1 x_Methanizer_prod_Wheat_grain <= 0
_C7: - 0.6 N_Methanizer_waste - delta_Methanizer_2
 + 0.4 x_Methanizer_excr_bovines_manure + 0.4 x_Methanizer_excr_bovines_slurry
 + 0.4 x_Methanizer_excr_porcines_manure
 + 0.4 x_Methanizer_excr_porcines_slurry - 0.6 x_Methanizer_prod_Barley_grain
 - 0.6 x_Methanizer_prod_Forage - 0.6 x_Methanizer_prod_Maize_corn
 - 0.6 x_Methanizer_prod_Natural_meadow_forage
 - 0.6 x_Methanizer_prod_Oats_grain - 0.6 x_Methanizer_prod_Wheat_grain <= 0
_C8: 0.6 N_Methanizer_waste - delta_Methanizer_2
 - 0.4 x_Methanizer_excr_bovines_manure - 0.4 x_Methanizer_excr_bovines_slurry
 - 0.4 x_Methanizer_excr_porcines_manure
 - 0.4 x_Methanizer_excr_porcines_slurry + 0.6 x_Methanizer_prod_Barley_grain
 + 0.6 x_Methanizer_prod_Forage + 0.6 x_Methanizer_prod_Maize_corn
 + 0.6 x_Methanizer_prod_Natural_meadow_forage
 + 0.6 x_Methanizer_prod_Oats_grain + 0.6 x_Methanizer_prod_Wheat_grain <= 0
_C9: - 0.1 N_Methanizer_waste - delta_Methanizer_3
 - 0.1 x_Methanizer_excr_bovines_manure - 0.1 x_Methanizer_excr_bovines_slurry
 - 0.1 x_Methanizer_excr_porcines_manure
 - 0.1 x_Methanizer_excr_porcines_slurry - 0.1 x_Methanizer_prod_Barley_grain
 + 0.9 x_Methanizer_prod_Forage - 0.1 x_Methanizer_prod_Maize_corn
 + 0.9 x_Methanizer_prod_Natural_meadow_forage
 - 0.1 x_Methanizer_prod_Oats_grain - 0.1 x_Methanizer_prod_Wheat_grain <= 0
Binaries
y_Barley_grain
y_Forage
y_Maize_corn
y_Meat_of_cattle_with_the_bone_fresh_or_chilled
y_Meat_of_pig_with_the_bone_fresh_or_chilled
y_Natural_meadow_forage
y_Non_edible_porcines_carcass
y_Oats_grain
y_Raw_milk_of_cattle
y_Rice_grain
y_Soya_beans_grain
y_Wheat_grain
End
