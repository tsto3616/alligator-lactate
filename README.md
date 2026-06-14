# alligator-lactate
The repository provides code for the American alligator hyperlactatemia modelling.

The details the general description and instructions for simulations. Data is available in either the python scripts or in the compartments.xlsx spreadsheet. The compartments.xlsx spreasheet contains three sheets, the first sheet "metabolites" details the metabolite concentrations, the "enzymes" sheet details the enzymatic activities of the enzymes and the "meta-enzymes" sheet details the enzymatic pathways with a 1:1 stereochemistry for each reaction. The sample sizes, metabolites/ enzymatic activities and or standard errors are featured for the enzymes and metabolites under the columns: "n", "mean" "se". The models are dependent on the compartments of organs for which the enzymes and metabolites are considered separate. The compartments are the blood, liver, kidneys and muscle. Under simulation one the meta-enzymes and enzymes sheet lacked the rows for aspartate aminotransferase - so to reproduce simulation A remove those rows. For simulation A and B the PEPCK forward and reverse reactions within meta-enzymes replaced the phosphoenolpyruvate with pyruvate under the assumption that phosphoenolpyruvate is metabolised into pyruvate anyways. The "study" column references the in-text referencing for the study (see the references of ...). 

All measurements are defined as either umol/g or umol/g/min under the assumption of crocodilian blood weighing 1060g/L. For all simulations the model assumes a respiratory quotient (RQ) of 85% or 0.85. This RQ is the rate of oxygen:carbon dioxide expiration and is supported by the literature, although this is a mean - not a fixed rate (1). The RQ was used to set the rate of anaerobic metabolism per energy demands. A standard metabolic rate was derived by 0.2 times by mass^0.83 as is standard for the determination of the standard metabolic weight of the American alligator (2). ATP was assumed to be directly proportional to the standard metabolic rate as it is commonly for ecological physiology studies (3). The Q10 quotient which dictates the change in the metabolic rate for an animal under increments of 10 degrees celsius can also be changed within the model to provide a more realistic model. All measurements were taken at approximately 30 degrees celsius and so the Q10 was kept to a value of 2.5 as is appropriate for crocodilians (4). The pKa dissociation constant for the rate of lactate to lactic acid dissociation is 3.86. In all simulations breathing was not included in the model due to the prolonged breath holding during dives by crocodilians and the ability to utilise almost all available oxygen in the blood (5). The relative weights of the organsa are 6%, 40%, 0.3% and 5% of the blood, liver, kidneys and muscle. The body weight was 1.5kg although the weight can be altered easily (instructions for model tweaking are available in the script). The model was simulated for 240 minutes/ 3 hours although this can be tweaked too. 

**Simulation A:**
Simulation A uses the self-tuning hyperparameter for ammonium bicarbonate excretion using the assumption that the initial blood sampling featured in the spreadsheet is representative of the 'truth' and will remain the same throughout the model. Without this hyperparameterisation the blood pH would far exceed what is physiologically possible. The hyperparameter was self-tuned for all four metabolic states (5%, 10%, 50% and 100% anaerobic metabolism). Further details for the simulation A are contained within the script for simulation A. 

**Simulation B and C**
Both simulation B and C use the same code. The only difference is the inclusion of the pyruvate from PEPCK metabolism for simulation C. The model has multiple self-tuning hyperparameters which will be described in greater detail below:

Hyperparameter 1: Scaling of the proton production from anaerobic metabolism. 

Hyperparameter 2: Rate constant of the excretion of ammonium. 

Hyperparameter 3: Enzymatic activity rate of carbonic anhydrase. 

Hyperparameter 4: Determines the strength of the inhibitory effects of dropping pH on the aerobic respiration. 

Hyperparameter 5: Determines the strength of the inhibitory effects of carbon dioxide upon aerobic respiration. 

Hyperparameter 6: Determines the strength of the hyperparameter 4 and 5. 

Hyperparameter 7: Determines the midpoint at which the proton:carbon dioxide balance is predictive of 50% anaerobic respiration. 

Hyperparameter 8: Determined the aspartate dose needed to achieve the minimal anaerobic:aerobic respiration rate. 

Hyperparameter 9: Determined the effectiveness of aspartate upon the ammonium excretion. The effect is given as a percentage in the study by the formula = rate of kidney excretion of NH4(1 + hyperparameter 9). 

Hyperparameter 10: Determined the effectiveness of aspartate upon the aerobic metabolism fraction. The effect is given as a percentage in the study by the formula = 0.5/(1 + hyperparameter 10) where 0.5 is the half way point between anaerobic and aerobic metabolism.  

The full detail is in the script. 

1. Smith EN. Oxygen Consumption, Ventilation, and Oxygen Pulse of the American Alligator during Heating and Cooling. Physiological Zoology. 1975;48(4):326-37.
2. Seymour RS, Gienger CM, Brien ML, Tracy CR, Charlie Manolis S, Webb GJ, et al. Scaling of standard metabolic rate in estuarine crocodiles Crocodylus porosus. J Comp Physiol B. 2013;183(4):491-500.
3. Killen SS, Christensen EAF, Cortese D, Zavorka L, Norin T, Cotgrove L, et al. Guidelines for reporting methods to estimate metabolic rates by aquatic intermittent-flow respirometry. J Exp Biol. 2021;224(18).
4. Franklin CE, Seebacher F. The effect of heat transfer mode on heart rate responses and hysteresis during heating and cooling in the estuarine crocodile Crocodylus porosus. J Exp Biol. 2003;206(Pt 7):1143-51.
5. Takahashi K, Lee Y, Fago A, Bautista NM, Storz JF, Kawamoto A, et al. The unique allosteric property of crocodilian haemoglobin elucidated by cryo-EM. Nat Commun. 2024;15(1):6505.
