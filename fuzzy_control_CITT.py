"""
Fecha modificacion: 29/05/2018
-------------------------------------------------------------

"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from scipy.stats import beta
from random import randrange
from random import uniform
import csv
import time


start = time.time()

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rcParams.update({'font.size': BIGGER_SIZE})
#RANDOM_SEED = 42

########## PATH results ########################

path="/experiments/"

############### simulacion vectors ########################################
CPU_SIM=[]
BATTERY_SIM=[]
TEMPERATURE_SIM=[]
RESPONSE_TIME_SIM=[]
output_sim=[]

#############################################################
################ vector generation function #################

###variable="CPU", "BATTERY","TEMPERATURE","RESPONSE-TIME"
####state="poor","normal","optimum"
###number=number of values to generate

def generate_vector(variable,state,number):
	#random.seed(RANDOM_SEED)
	output_sim=[]
	if variable=="CPU":
		if state=="optimum":
			for i in range(0,number):
				output_sim.append(randrange(0,30))
		elif state=="normal":
			for i in range(0,number):
				output_sim.append(randrange(20,70))
		elif state=="poor":
			for i in range(0,number):
				output_sim.append(randrange(60,100)) # max 100
		else:
			print("ERROR CPU STATE NOT DEFINED")
	elif variable=="BATTERY":
		if state=="optimum":
			for i in range(0,number):
				output_sim.append(randrange(70,100)) # max 100
		elif state=="normal":
			for i in range(0,number):
				output_sim.append(randrange(30,80))
		elif state=="poor":
			for i in range(0,number):
				output_sim.append(randrange(0,40))
		else:
			print("ERROR BATTERY STATE NOT DEFINED")
	
	elif variable=="TEMPERATURE":
		if state=="poor":
			for i in range(0,number):
				output_sim.append(randrange(30,40))
		elif state=="normal":
			for i in range(0,number):
				output_sim.append(randrange(20,35))
		elif state=="optimum":
			for i in range(0,number):
				output_sim.append(randrange(0,25))
		else:
			print("ERROR TEMPERATURE STATE NOT DEFINED")
	

	elif variable=="RESPONSE_TIME":
		if state=="optimum":
			for i in range(0,number):
				output_sim.append(randrange(0,300))
		elif state=="normal":
			for i in range(0,number):
				output_sim.append(randrange(250,1500))
		elif state=="poor":
			for i in range(0,number):
				output_sim.append(randrange(1400,4000))
		else:
			print("ERROR RESPONSE_TIME STATE NOT DEFINED")
	else:											
		print("ERROR VARIABLE NOT DEFINED")


				
	return output_sim

####################################################
########VALUES(MAX,MIN,STEP)########################

battery_universe=[0,100,10]
temp_universe=[0,40,5]
cpu_util_universe=[0,100,10]
response_time_universe=[0,4000,20]
output_fail_universe=[0,100,5]   

#######################################################
# New Antecedent/Consequent objects hold universe variables and membership
# functions

############################INPUTS###################################################################################
battery = ctrl.Antecedent(np.arange(battery_universe[0], battery_universe[1], battery_universe[2]), 'Battery charge level')
temp = ctrl.Antecedent(np.arange(temp_universe[0], temp_universe[1], temp_universe[2]), 'Battery temperature')
cpu_util=ctrl.Antecedent(np.arange(cpu_util_universe[0], cpu_util_universe[1], cpu_util_universe[2]),'CPU utilization')
response_time=ctrl.Antecedent(np.arange(response_time_universe[0],response_time_universe[1],response_time_universe[2]),'Response time')
#######################################################################################################################

##################OUTPUT#################################################
output_fail = ctrl.Consequent(np.arange(output_fail_universe[0], output_fail_universe[1], output_fail_universe[2]), 'Failure-risk')
#########################################################################

battery['poor'] = fuzz.trapmf(battery.universe, [0, 0, 20, 40])
battery['normal'] = fuzz.trapmf(battery.universe, [30, 35, 75, 80])
battery['optimum'] = fuzz.trapmf(battery.universe, [70, 80, 90, 100])

temp['optimum'] = fuzz.trapmf(temp.universe, [0, 0, 20, 25])
temp['normal'] = fuzz.trapmf(temp.universe, [20, 25, 30, 35])
temp['poor'] = fuzz.trapmf(temp.universe, [30, 35, 40, 40])

cpu_util['optimum'] = fuzz.trapmf(cpu_util.universe, [0, 0, 20, 30])
cpu_util['normal'] = fuzz.trapmf(cpu_util.universe, [20, 30, 60, 70])
cpu_util['poor'] = fuzz.trapmf(cpu_util.universe, [60, 70, 90, 100])

response_time['optimum'] = fuzz.trapmf(response_time.universe, [0, 0, 100, 300])
response_time['normal'] = fuzz.trapmf(response_time.universe, [250,500,1200,1500])
response_time['poor'] = fuzz.trapmf(response_time.universe, [1400, 1500, 4000, 4000])


# Custom membership functions can be built interactively with a familiar,
# Pythonic API
output_fail['very-low'] = fuzz.trapmf(output_fail.universe, [0,0,17,23])
output_fail['low'] = fuzz.trapmf(output_fail.universe, [16,24,56,64])
output_fail['high'] = fuzz.trapmf(output_fail.universe, [57,63,86,92])
output_fail['very-high'] = fuzz.trapmf(output_fail.universe, [88,90,100,100])
"""

To help understand what the membership looks like, use the ``view`` methods.
"""
# You can see how these look with .view()
battery.view()
temp.view()
cpu_util.view()
response_time.view()
output_fail.view()

"""
Fuzzy rules
"""
###########################################################################################################################

rule1 = ctrl.Rule(temp['poor'] & battery['poor'] & cpu_util['poor'] & response_time['poor'], output_fail['very-high'])
rule2 = ctrl.Rule(temp['poor'] & battery['poor'] & cpu_util['poor'] & response_time['normal'], output_fail['very-high'])
rule3 = ctrl.Rule(temp['poor'] & battery['poor'] & cpu_util['poor'] & response_time['optimum'], output_fail['very-high'])


rule4 = ctrl.Rule(temp['poor'] & battery['poor'] & cpu_util['normal'] & response_time['poor'], output_fail['very-high'])
rule5 = ctrl.Rule(temp['poor'] & battery['poor'] & cpu_util['normal'] & response_time['normal'], output_fail['high'])
rule6 = ctrl.Rule(temp['poor'] & battery['poor'] & cpu_util['normal'] & response_time['optimum'], output_fail['high'])


rule7 = ctrl.Rule(temp['poor'] & battery['poor'] & cpu_util['optimum'] & response_time['poor'], output_fail['very-high'])
rule8 = ctrl.Rule(temp['poor'] & battery['poor'] & cpu_util['optimum'] & response_time['normal'], output_fail['high'])
rule9 = ctrl.Rule(temp['poor'] & battery['poor'] & cpu_util['optimum'] & response_time['optimum'], output_fail['high'])

####################################################################################################################################
rule10 = ctrl.Rule(temp['poor'] & battery['normal'] & cpu_util['poor'] & response_time['poor'], output_fail['very-high'])
rule11 = ctrl.Rule(temp['poor'] & battery['normal'] & cpu_util['poor'] & response_time['normal'], output_fail['high'])
rule12 = ctrl.Rule(temp['poor'] & battery['normal'] & cpu_util['poor'] & response_time['optimum'], output_fail['high'])


rule13 = ctrl.Rule(temp['poor'] & battery['normal'] & cpu_util['normal'] & response_time['poor'], output_fail['high'])
rule14 = ctrl.Rule(temp['poor'] & battery['normal'] & cpu_util['normal'] & response_time['normal'], output_fail['low'])
rule15 = ctrl.Rule(temp['poor'] & battery['normal'] & cpu_util['normal'] & response_time['optimum'], output_fail['low'])


rule16 = ctrl.Rule(temp['poor'] & battery['normal'] & cpu_util['optimum'] & response_time['poor'], output_fail['high'])
rule17 = ctrl.Rule(temp['poor'] & battery['normal'] & cpu_util['optimum'] & response_time['normal'], output_fail['low'])
rule18 = ctrl.Rule(temp['poor'] & battery['normal'] & cpu_util['optimum'] & response_time['optimum'], output_fail['low'])

###################################################################################################################################

rule19 = ctrl.Rule(temp['poor'] & battery['optimum'] & cpu_util['poor'] & response_time['poor'], output_fail['very-high'])
rule20 = ctrl.Rule(temp['poor'] & battery['optimum'] & cpu_util['poor'] & response_time['normal'], output_fail['high'])
rule21 = ctrl.Rule(temp['poor'] & battery['optimum'] & cpu_util['poor'] & response_time['optimum'], output_fail['high'])


rule22 = ctrl.Rule(temp['poor'] & battery['optimum'] & cpu_util['normal'] & response_time['poor'], output_fail['high'])
rule23 = ctrl.Rule(temp['poor'] & battery['optimum'] & cpu_util['normal'] & response_time['normal'], output_fail['low'])
rule24 = ctrl.Rule(temp['poor'] & battery['optimum'] & cpu_util['normal'] & response_time['optimum'], output_fail['low'])


rule25 = ctrl.Rule(temp['poor'] & battery['optimum'] & cpu_util['optimum'] & response_time['poor'], output_fail['high'])
rule26 = ctrl.Rule(temp['poor'] & battery['optimum'] & cpu_util['optimum'] & response_time['normal'], output_fail['low'])
rule27 = ctrl.Rule(temp['poor'] & battery['optimum'] & cpu_util['optimum'] & response_time['optimum'], output_fail['low'])

######################################################################################################################################
#####################################################################################################################################

rule28 = ctrl.Rule(temp['normal'] & battery['poor'] & cpu_util['poor'] & response_time['poor'], output_fail['very-high'])
rule29 = ctrl.Rule(temp['normal'] & battery['poor'] & cpu_util['poor'] & response_time['normal'], output_fail['high'])
rule30 = ctrl.Rule(temp['normal'] & battery['poor'] & cpu_util['poor'] & response_time['optimum'], output_fail['high'])


rule31 = ctrl.Rule(temp['normal'] & battery['poor'] & cpu_util['normal'] & response_time['poor'], output_fail['high'])
rule32 = ctrl.Rule(temp['normal'] & battery['poor'] & cpu_util['normal'] & response_time['normal'], output_fail['low'])
rule33 = ctrl.Rule(temp['normal'] & battery['poor'] & cpu_util['normal'] & response_time['optimum'], output_fail['low'])


rule34 = ctrl.Rule(temp['normal'] & battery['poor'] & cpu_util['optimum'] & response_time['poor'], output_fail['high'])
rule35 = ctrl.Rule(temp['normal'] & battery['poor'] & cpu_util['optimum'] & response_time['normal'], output_fail['low'])
rule36 = ctrl.Rule(temp['normal'] & battery['poor'] & cpu_util['optimum'] & response_time['optimum'], output_fail['low'])

####################################################################################################################################
rule37 = ctrl.Rule(temp['normal'] & battery['normal'] & cpu_util['poor'] & response_time['poor'], output_fail['high'])
rule38 = ctrl.Rule(temp['normal'] & battery['normal'] & cpu_util['poor'] & response_time['normal'], output_fail['low'])
rule39 = ctrl.Rule(temp['normal'] & battery['normal'] & cpu_util['poor'] & response_time['optimum'], output_fail['low'])


rule40 = ctrl.Rule(temp['normal'] & battery['normal'] & cpu_util['normal'] & response_time['poor'], output_fail['low'])
rule41 = ctrl.Rule(temp['normal'] & battery['normal'] & cpu_util['normal'] & response_time['normal'], output_fail['very-low'])
rule42 = ctrl.Rule(temp['normal'] & battery['normal'] & cpu_util['normal'] & response_time['optimum'], output_fail['very-low'])


rule43 = ctrl.Rule(temp['normal'] & battery['normal'] & cpu_util['optimum'] & response_time['poor'], output_fail['low'])
rule44 = ctrl.Rule(temp['normal'] & battery['normal'] & cpu_util['optimum'] & response_time['normal'], output_fail['very-low'])
rule45 = ctrl.Rule(temp['normal'] & battery['normal'] & cpu_util['optimum'] & response_time['optimum'], output_fail['very-low'])

###################################################################################################################################

rule46 = ctrl.Rule(temp['normal'] & battery['optimum'] & cpu_util['poor'] & response_time['poor'], output_fail['high'])
rule47 = ctrl.Rule(temp['normal'] & battery['optimum'] & cpu_util['poor'] & response_time['normal'], output_fail['low'])
rule48 = ctrl.Rule(temp['normal'] & battery['optimum'] & cpu_util['poor'] & response_time['optimum'], output_fail['low'])


rule49 = ctrl.Rule(temp['normal'] & battery['optimum'] & cpu_util['normal'] & response_time['poor'], output_fail['low'])
rule50 = ctrl.Rule(temp['normal'] & battery['optimum'] & cpu_util['normal'] & response_time['normal'], output_fail['very-low'])
rule51 = ctrl.Rule(temp['normal'] & battery['optimum'] & cpu_util['normal'] & response_time['optimum'], output_fail['very-low'])


rule52 = ctrl.Rule(temp['normal'] & battery['optimum'] & cpu_util['optimum'] & response_time['poor'], output_fail['low'])
rule53 = ctrl.Rule(temp['normal'] & battery['optimum'] & cpu_util['optimum'] & response_time['normal'], output_fail['very-low'])
rule54 = ctrl.Rule(temp['normal'] & battery['optimum'] & cpu_util['optimum'] & response_time['optimum'], output_fail['very-low'])

######################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

rule55 = ctrl.Rule(temp['optimum'] & battery['poor'] & cpu_util['poor'] & response_time['poor'], output_fail['very-high'])
rule56 = ctrl.Rule(temp['optimum'] & battery['poor'] & cpu_util['poor'] & response_time['normal'], output_fail['high'])
rule57 = ctrl.Rule(temp['optimum'] & battery['poor'] & cpu_util['poor'] & response_time['optimum'], output_fail['high'])


rule58 = ctrl.Rule(temp['optimum'] & battery['poor'] & cpu_util['normal'] & response_time['poor'], output_fail['high'])
rule59 = ctrl.Rule(temp['optimum'] & battery['poor'] & cpu_util['normal'] & response_time['normal'], output_fail['low'])
rule60 = ctrl.Rule(temp['optimum'] & battery['poor'] & cpu_util['normal'] & response_time['optimum'], output_fail['low'])


rule61 = ctrl.Rule(temp['optimum'] & battery['poor'] & cpu_util['optimum'] & response_time['poor'], output_fail['high'])
rule62 = ctrl.Rule(temp['optimum'] & battery['poor'] & cpu_util['optimum'] & response_time['normal'], output_fail['low'])
rule63 = ctrl.Rule(temp['optimum'] & battery['poor'] & cpu_util['optimum'] & response_time['optimum'], output_fail['low'])

####################################################################################################################################
rule64 = ctrl.Rule(temp['optimum'] & battery['normal'] & cpu_util['poor'] & response_time['poor'], output_fail['high'])
rule65 = ctrl.Rule(temp['optimum'] & battery['normal'] & cpu_util['poor'] & response_time['normal'], output_fail['low'])
rule66 = ctrl.Rule(temp['optimum'] & battery['normal'] & cpu_util['poor'] & response_time['optimum'], output_fail['low'])


rule67 = ctrl.Rule(temp['optimum'] & battery['normal'] & cpu_util['normal'] & response_time['poor'], output_fail['low'])
rule68 = ctrl.Rule(temp['optimum'] & battery['normal'] & cpu_util['normal'] & response_time['normal'], output_fail['very-low'])
rule69 = ctrl.Rule(temp['optimum'] & battery['normal'] & cpu_util['normal'] & response_time['optimum'], output_fail['very-low'])


rule70 = ctrl.Rule(temp['optimum'] & battery['normal'] & cpu_util['optimum'] & response_time['poor'], output_fail['low'])
rule71 = ctrl.Rule(temp['optimum'] & battery['normal'] & cpu_util['optimum'] & response_time['normal'], output_fail['very-low'])
rule72 = ctrl.Rule(temp['optimum'] & battery['normal'] & cpu_util['optimum'] & response_time['optimum'], output_fail['very-low'])

###################################################################################################################################

rule73 = ctrl.Rule(temp['optimum'] & battery['optimum'] & cpu_util['poor'] & response_time['poor'], output_fail['high'])
rule74 = ctrl.Rule(temp['optimum'] & battery['optimum'] & cpu_util['poor'] & response_time['normal'], output_fail['low'])
rule75 = ctrl.Rule(temp['optimum'] & battery['optimum'] & cpu_util['poor'] & response_time['optimum'], output_fail['low'])


rule76 = ctrl.Rule(temp['optimum'] & battery['optimum'] & cpu_util['normal'] & response_time['poor'], output_fail['low'])
rule77 = ctrl.Rule(temp['optimum'] & battery['optimum'] & cpu_util['normal'] & response_time['normal'], output_fail['very-low'])
rule78 = ctrl.Rule(temp['optimum'] & battery['optimum'] & cpu_util['normal'] & response_time['optimum'], output_fail['very-low'])


rule79 = ctrl.Rule(temp['optimum'] & battery['optimum'] & cpu_util['optimum'] & response_time['poor'], output_fail['low'])
rule80 = ctrl.Rule(temp['optimum'] & battery['optimum'] & cpu_util['optimum'] & response_time['normal'], output_fail['very-low'])
rule81 = ctrl.Rule(temp['optimum'] & battery['optimum'] & cpu_util['optimum'] & response_time['optimum'], output_fail['very-low'])

######################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

#rule1.view()

"""
.. image:: PLOT2RST.current_figure

Control System Creation and Simulation
---------------------------------------

Now that we have our rules defined, we can simply create a control system
via:
"""

fail_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4, rule5, rule6,rule7, rule8,rule9,rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27,rule28,rule29,rule30,rule31,rule32,rule33,rule34,rule35,rule36,rule37,rule38,rule39,rule40,rule41,rule42,rule43,rule44,rule45,rule46,rule47,rule48,rule49,rule50,rule51,rule52,rule53,rule54,rule55,rule56,rule57,rule58,rule59,rule60,rule61,rule62,rule63,rule64,rule65,rule66,rule67,rule68,rule69,rule70,rule71,rule72,rule73,rule74,rule75,rule76,rule77,rule78,rule79,rule80,rule81])

"""
In order to simulate this control system, we will create a
``ControlSystemSimulation``.  Think of this object representing our controller
applied to a specific set of cirucmstances.  For tipping, this might be tipping
Sharon at the local brew-pub.  We would create another
``ControlSystemSimulation`` when we're trying to apply our ``tipping_ctrl``
for Travis at the cafe because the inputs would be different.
"""

fail = ctrl.ControlSystemSimulation(fail_ctrl)

"""
We can now simulate our control system by simply specifying the inputs
and calling the ``compute`` method.  Suppose we rated the quality 6.5 out of 10
and the service 9.8 of 10.
"""
# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)

##########################################################
################## SIMULATION#############################
##########################################################

levels=["poor","normal","optimum"]
variables=["CPU","BATTERY","TEMPERATURE","RESPONSE_TIME"]
output_tags=["very-low","low","high","very-high"]


TEMPERATURE_SIM=generate_vector("TEMPERATURE","normal",1)
print("TEMPERATURE: ",TEMPERATURE_SIM)

BATTERY_SIM=generate_vector("BATTERY","poor",1)
print("BATTERY: ",BATTERY_SIM)

RESPONSE_TIME_SIM=generate_vector("RESPONSE_TIME","poor",1)

CPU_SIM=generate_vector("CPU","poor",1)
print("CPU: ",CPU_SIM)

print("RESPONSE_TIME: ",RESPONSE_TIME_SIM)

###############################################################################

output=[]
###############################################################################

######## CSV Header: File Create ###########
def crete_header(filename):
	with open(filename,'w') as csvfile:
		fieldnames=['id','TEMPERATURE','BATTERY','CPU','RESPONSE_TIME','OUTPUT']
		writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
		writer.writeheader()
		#writer.close()
		return
###############################################################################
################################## Write  CSV #################################

def write_csv(filename,data):
	with open(filename, 'at') as csvfile:
		fieldnames=['id','TEMPERATURE','BATTERY','CPU','RESPONSE_TIME','OUTPUT']
		writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
		writer.writerow(data)
		#writer.close()
		print("writing complete")
		return
##############################################################################
############################# File Name ######################################

######################## SIMULATIONS ########################################

#file_name="experiments-very-low.csv"

#file_name="experiments-poor-temp.csv"
#file_name="experiments-poor-battery.csv"
#file_name="experiments-poor-respons.csv"
#file_name="experiments-poor-cpu.csv"


#file_name="experiments-high-temp-battery.csv"
#file_name="experiments-high-battery-respons.csv"
#file_name="experiments-high-respons-cpu.csv"
#file_name="experiments-high-temp-cpu.csv"
#file_name="experiments-high-temp-response.csv"
#file_name="experiments-high-battery-cpu.csv"


file_name="experiments-veryhigh-temp-battery-respons-cpu.csv"
#file_name="experiments-veryhigh-temp-battery-respons.csv"
#file_name="experiments-veryhigh-temp-respons-cpu.csv"
#file_name="experiments-veryhigh-temp-battery-cpu.csv"
#file_name="experiments-veryhigh-battery-respons-cpu.csv"


crete_header(file_name)


for i in range(0,len(BATTERY_SIM)):

	fail.input['Battery charge level'] = BATTERY_SIM[i]
	fail.input['Battery temperature']=TEMPERATURE_SIM[i]
	fail.input['CPU utilization']=CPU_SIM[i]
	fail.input['Response time']=RESPONSE_TIME_SIM[i]
# Crunch the numbers
	fail.compute()
	output.append(fail.output)
	print(fail.output['Failure-risk'])
	data={"id":i,"TEMPERATURE":TEMPERATURE_SIM[i],"BATTERY":BATTERY_SIM[i],"CPU":CPU_SIM[i],"RESPONSE_TIME":RESPONSE_TIME_SIM[i],"OUTPUT":fail.output['Failure-risk']}
	print("Battery-Level:%s,Temperature:%s,CPU-Utilization:%s,Response-Time:%s,OUTPUT:%s"%(BATTERY_SIM[i],TEMPERATURE_SIM[i],CPU_SIM[i],RESPONSE_TIME_SIM[i],fail.output['Failure-risk'])) 
	write_csv(file_name,data)
	output_fail.view(sim=fail)
	#plt.show()

end = time.time()
print("elapsed time")
print(end - start)
	#plt.savefig("images/"+file_name+"_"+str(i)+".pdf")

##############################################################################

"""
Once computed, we can view the result as well as visualize it.
"""
#print (fail.output)
#output_fail.view(sim=fail)
#plt.show()
"""
.. image:: PLOT2RST.current_figure

The resulting suggested tip is **20.24%**.

Final thoughts
--------------

The power of fuzzy systems is alpooring complicated, intuitive behavior based
on a sparse system of rules with minimal overhead. Note our membership
function universes were coarse, only defined at the integers, but
``fuzz.interp_membership`` alpoored the effective resolution to increase on
demand. This system can respond to arbitrarily small changes in inputs,
and the processing burden is minimal.

"""
