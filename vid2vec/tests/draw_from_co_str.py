# Copy this string from .CSV
# Original
# str = "827	549	-110.87603	833	596	-122.47882	847	643	-133.81035	860	676	-138.51346	880	716	-127.4393	907	763	-97.15387	933	790	-56.081615	973	823	-16.222385	1027	836	-3.2471075	1080	816	-28.28373	1113	790	-73.01176	1134	763	-119.5503	1160	723	-154.2458	1180	683	-166.9389	1194	643	-164.0253	1207	596	-155.2223	1220	543	-144.28827	880	516	70.289795	913	503	99.857834	947	496	116.42525	973	503	123.99855	1000	516	123.78034	1100	516	116.75095	1127	509	112.42443	1154	503	99.949265	1180	509	78.673645	1194	523	44.87882	1047	576	103.788506	1047	616	113.784424	1047	656	128.81512	1047	683	125.31224	1007	690	76.7296	1020	696	82.47478	1040	696	83.82086	1060	696	79.01868	1073	690	71.56852	920	563	75.85306	940	556	95.25523	967	556	93.68959	993	569	80.78839	967	576	88.10806	940	569	86.30222	1100	569	75.640785	1120	556	83.7421	1147	556	80.32187	1160	569	56.690674	1140	576	70.72013	1120	576	78.10249	973	736	40.303738	993	736	66.50421	1027	730	81.45837	1040	736	81.71853	1053	730	79.358955	1080	736	59.48197	1100	743	28.185108	1080	763	50.119175	1060	776	61.45346	1040	776	65.76784	1013	776	65.24269	993	763	57.58033	973	736	40.2734	1020	743	67.53007	1040	750	70.223946	1060	750	64.61998	1093	743	29.310999	1060	750	63.25984	1040	756	67.00127	1020	750	66.16734"

# PB
str = "862.6181247	502.4949012	-31.81727863	872.2574898	549.0366959	-42.79765863	889.8229904	595.1435924	-52.74807577	904.9988044	627.36055	-56.19503843	926.0195607	666.2673314	-43.27114051	952.5595502	711.8818078	-10.55790431	975.9867174	737.636998	32.79838799	1013.773419	768.6133217	76.27701237	1066.930046	778.7162315	94.3231209	1120.890094	755.7254211	74.43422276	1156.509456	727.7319185	33.04314501	1180.317959	699.3849603	-11.29015175	1207.260863	657.8464669	-43.35900651	1226.152981	616.749711	-54.09489432	1237.600935	576.061408	-49.86459335	1247.112803	528.4688067	-39.86652506	1256.093827	474.8963073	-27.74705055	896.3080331	467.5967364	153.5638538	925.5927761	452.9703614	186.1328868	957.4333059	444.2127782	205.8550718	982.9420565	449.8236709	215.8638903	1010.512519	461.3304648	218.2114201	1110.577547	455.8400094	220.7128969	1137.441946	447.3554608	218.9706894	1165.134016	439.8270032	209.1166854	1193.324246	444.289262	190.4070316	1211.212395	457.3288177	158.0948913	1062.413098	518.5729884	202.774551	1063.656463	558.564933	212.7252692	1064.422288	598.5830797	227.6880033	1066.233764	625.5242984	224.2009589	1031.46592	634.4425	172.0383689	1044.171499	639.7543842	178.9924486	1063.923486	638.6706149	182.2322623	1084.258625	637.5548487	179.3516079	1097.55832	630.8160845	173.1700196	938.1146534	512.37355	162.9015923	955.7705488	504.3942617	184.1158504	982.7565891	502.9135681	185.1220381	1010.536027	514.4088957	174.7489367	984.3817159	522.8544825	179.5657492	957.3319491	517.3281436	175.203324	1117.380452	508.5464624	179.7886224	1135.779494	494.5173735	189.7531237	1162.941439	493.0270282	188.9131063	1178.816817	505.1755182	166.6236482	1157.989994	513.3287917	178.6898504	1137.41013	514.4579857	184.1390149	1003.645599	682.0381614	132.5475328	1021.040194	681.0837395	160.5293479	1053.088525	673.3162587	178.6455694	1066.314347	678.5995977	180.1394357	1079.131209	671.8873259	179.0254123	1108.182765	676.3023248	161.8030608	1131.41438	682.0381614	132.5475328	1110.550043	703.2130473	152.4826012	1090.307571	717.3432832	161.8658191	1070.018699	718.4565107	164.2608711	1043.230967	719.9263234	161.1733478	1023.365842	707.9967463	151.6458188	1003.648477	682.0380035	132.5173317	1048.163941	686.6060192	164.1153166	1068.171598	692.5187508	168.6968293	1088.582784	691.398812	165.0180197	1124.34971	682.4257918	133.0033969	1088.711791	691.3917335	163.6640283	1068.805982	698.4929678	165.4887227	1048.676698	693.588414	162.7587543"


import matplotlib.pyplot as plt

data = [float(x) for x in str.split("\t")]

# print(data)

landmarks = []
X = []
Y = []
for i in range(1,69):
  x,y,z = data.pop(0),data.pop(0),data.pop(0)
  

  X.append(x)
  Y.append(y)
  landmarks.append((x,y))

# print(landmarks)
plt.scatter(X,Y,color='black')
plt.show()