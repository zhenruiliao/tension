Search.setIndex({docnames:["examples","index","installation","layers","models","notes","usage"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["examples.md","index.rst","installation.rst","layers.rst","models.rst","notes.rst","usage.rst"],objects:{"base.FORCELayer":[[3,1,1,"","build"],[3,1,1,"","call"],[3,1,1,"","from_weights"],[3,1,1,"","get_initial_state"],[3,1,1,"","initialize_feedback_kernel"],[3,1,1,"","initialize_input_kernel"],[3,1,1,"","initialize_output_kernel"],[3,1,1,"","initialize_recurrent_kernel"]],"base.FORCEModel":[[4,1,1,"","build"],[4,1,1,"","call"],[4,1,1,"","coerce_input_data"],[4,1,1,"","compile"],[4,1,1,"","evaluate"],[4,1,1,"","fit"],[4,1,1,"","force_layer_call"],[4,1,1,"","initialize_P"],[4,1,1,"","initialize_train_idx"],[4,1,1,"","predict"],[4,1,1,"","pseudogradient_P"],[4,1,1,"","pseudogradient_P_Gx"],[4,1,1,"","pseudogradient_wO"],[4,1,1,"","pseudogradient_wR"],[4,1,1,"","train_step"],[4,1,1,"","update_output_kernel"],[4,1,1,"","update_recurrent_kernel"]],"constrained.BioFORCEModel":[[4,1,1,"","pseudogradient_wR"]],"constrained.ConstrainedNoFeedbackESN":[[3,1,1,"","build"],[3,1,1,"","call"],[3,1,1,"","from_weights"],[3,1,1,"","get_initial_state"],[3,1,1,"","initialize_recurrent_kernel"]],"models.EchoStateNetwork":[[3,1,1,"","call"],[3,1,1,"","get_initial_state"]],"models.FullFORCEModel":[[4,1,1,"","build"],[4,1,1,"","call"],[4,1,1,"","force_layer_call"],[4,1,1,"","initialize_P"],[4,1,1,"","initialize_target_network"],[4,1,1,"","initialize_train_idx"],[4,1,1,"","pseudogradient_wR_task"],[4,1,1,"","update_recurrent_kernel"]],"models.NoFeedbackESN":[[3,1,1,"","build"],[3,1,1,"","call"],[3,1,1,"","from_weights"]],"models.OptimizedFORCEModel":[[4,1,1,"","initialize_P"],[4,1,1,"","pseudogradient_P_Gx"],[4,1,1,"","pseudogradient_wR"]],"spiking.Izhikevich":[[3,1,1,"","initialize_voltage"],[3,1,1,"","update_voltage"]],"spiking.LIF":[[3,1,1,"","initialize_voltage"],[3,1,1,"","update_voltage"]],"spiking.OptimizedSpikingNN":[[3,1,1,"","compute_current"],[3,1,1,"","update_firing_rate"]],"spiking.SpikingNN":[[3,1,1,"","call"],[3,1,1,"","compute_current"],[3,1,1,"","get_initial_state"],[3,1,1,"","initialize_feedback_kernel"],[3,1,1,"","initialize_output_kernel"],[3,1,1,"","initialize_recurrent_kernel"],[3,1,1,"","initialize_voltage"],[3,2,1,"","state_size"],[3,1,1,"","update_firing_rate"],[3,1,1,"","update_voltage"]],"spiking.SpikingNNModel":[[4,1,1,"","force_layer_call"]],"spiking.Theta":[[3,1,1,"","initialize_voltage"],[3,1,1,"","update_voltage"]],base:[[3,0,1,"","FORCELayer"],[4,0,1,"","FORCEModel"]],constrained:[[4,0,1,"","BioFORCEModel"],[3,0,1,"","ConstrainedNoFeedbackESN"]],models:[[3,0,1,"","EchoStateNetwork"],[4,0,1,"","FullFORCEModel"],[3,0,1,"","NoFeedbackESN"],[4,0,1,"","OptimizedFORCEModel"]],spiking:[[3,0,1,"","Izhikevich"],[3,0,1,"","LIF"],[3,0,1,"","OptimizedSpikingNN"],[3,0,1,"","SpikingNN"],[4,0,1,"","SpikingNNModel"],[3,0,1,"","Theta"]]},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"],"2":["py","property","Python property"]},objtypes:{"0":"py:class","1":"py:method","2":"py:property"},terms:{"0":[3,4,5,6],"1":[3,4,6],"2":[3,6],"25":[3,6],"2d":[3,4],"3":[2,3,6],"3d":4,"4":[3,6],"400":6,"5":[3,6],"50":6,"7":2,"abstract":3,"boolean":[3,6],"case":6,"class":[1,3,4],"default":[3,4,5,6],"float":[3,4],"function":[3,4,5,6],"import":6,"int":[3,4],"new":[1,2,3],"return":[3,4,6],"static":3,"super":6,"true":[3,4,6],"while":5,A:[1,3,4,6],As:5,By:[4,6],For:[4,6],If:[3,4,6],In:[5,6],It:3,NO:1,NOT:[],No:[],Not:6,One:[4,6],The:[3,4,6],To:[2,6],_:6,__init__:6,_g:6,_initial_a:6,_output_kernel_idx:6,_p_gg_idx:6,_p_output_idx:6,_p_recurr:6,_recurrent_kernel_idx:6,_recurrent_kernel_train:6,abbott:4,access:[1,4],accomod:6,action:3,activ:[1,2,3,5,6],ad:3,adapt:3,adapt_jump_curr:3,adapt_time_inv:3,add_weight:6,addit:[3,5],adjust:6,advanc:[],after:[4,5,6],al:[3,4,6],all:[4,5,6],allow:5,along:6,alpha_p:4,also:[3,4,6],altern:6,alwai:3,an:[3,4,5,6],ani:6,api_doc:3,appli:3,ar:[3,4,5,6],arg:[3,4],argument:4,around:4,arrai:[3,4,6],articl:6,assert:6,asset:[],assign_add:6,assum:6,attractor:1,attribut:1,auto:4,auxillari:[3,6],awai:5,backend:6,background:3,badg:[],base:[1,6],batch:[3,4,5,6],batch_siz:[3,4,6],behaviour:5,being:[4,6],below:6,bioforcemodel:[4,6],block:[],bool:[3,4],boolean_mask:6,build:[1,3,4],built:6,calcul:[],call:[3,4,6],callback:1,can:[3,4,6],capabl:6,capacit:3,cd:2,cell:3,chang:6,chao:3,chaotic:[1,6],cl:6,classmethod:[3,6],clone:2,clopath:[3,4],code:6,coerc:4,coerce_input_data:4,colab:[],com:2,compat:[1,4],compil:[4,6],compiled_metr:6,compon:4,comput:[3,4],compute_curr:3,concept:5,cond:6,conda:2,connect:[3,6],constant:[3,4,6],constrain:[3,4,6],constrainednofeedbackesn:[3,6],contain:[3,6],control:[1,3],convert_to_tensor:6,correspond:[3,4,6],count_nonzero:6,creat:[1,2,3],current:[3,6],custom:[1,4],custom_state_s:[],customizing_what_happens_in_fit:4,customspikingnnmodel:6,dadt:6,data:[1,4,6],decai:3,def:6,defin:6,definit:6,denser:3,densiti:[],depasqual:[4,6],depend:5,descent:5,describ:3,desir:[2,4,6],detail:[4,6],determinist:6,develop:1,deviat:3,differ:[4,6],dimens:[3,4,5,6],dimension:4,directori:2,divid:3,document:6,doe:4,dot:6,doubl:3,dp:4,dp_gx:4,dt:3,dtdivtau:[3,6],dtype:[3,6],durat:3,dure:[3,4,5,6],dwo:4,dwr:4,dynam:3,e:[2,3,6],each:[3,4,5,6],echo:[1,5],echostatenetwork:[3,6],els:6,end:4,environ:2,epoch:[4,6],equal:3,equiv:4,error:[1,4],esn:[3,4],et:[3,4,6],evalu:[4,5,6],everi:6,exampl:[1,4,5,6],exceed:[3,6],execut:4,exist:[4,6],experiment:5,exponenti:3,extern:6,factor:3,fals:[3,4,6],fb_hint_sum:4,feasibl:5,feedback:[3,4,6],feedback_kernel:[3,6],feedback_kernel_train:3,feedback_output_s:6,feedback_term:6,feedback_unit:6,ffmodel:6,fill:[],filter:3,find:4,fire:[3,4,6],first:[1,2,6],fit:[4,6],flatten:[5,6],follow:6,forc:[1,5,6],force_lay:[4,6],force_layer_cal:[1,4],forcelay:[1,3,4],forcemodel:[1,4],forward:[3,5,6],four:3,from:[3,4,5,6],from_weight:[1,3],full:[1,4,6],fullforc:6,fullforcemodel:[4,6],fulli:6,futur:3,g:3,gain:3,gain_on_v:3,gener:[4,6],get_initial_st:[3,6],git:2,github:2,githubtocolab:[],go:1,googl:[],gpu:1,gradient:5,gradual:5,ground:4,guid:[4,6],h:[3,4,6],h_target:4,h_task:4,ha:[3,6],hadjiabadi:[3,4],half:3,have:[3,4,6],here:[],hint:4,hint_dim:[4,6],histori:6,hph:4,hpht:4,hr:3,hr_ipsc:3,hscale:[3,6],http:[2,3,4],i:[3,6],i_bia:3,implement:[3,4,6],implicitli:[3,4],improv:3,includ:6,inconsist:6,indic:[3,4,6],individu:[5,6],infer:[3,4,5],inherit:[1,3],init_a:6,init_h:6,init_out:6,initi:[1,3,4,5],initial_a:[3,6],initial_h:3,initial_voltag:3,initialize_feedback_kernel:[3,6],initialize_input_kernel:[3,6],initialize_output_kernel:[3,6],initialize_p:4,initialize_recurrent_kernel:[3,6],initialize_target_network:4,initialize_train_idx:4,initialize_voltag:[3,6],input:[3,4,5,6],input_dim:[3,6],input_kernel:[3,6],input_kernel_train:3,input_shap:[3,4,6],input_tensor:[],input_term:6,input_unit:6,input_with_hint:[],inputs_with_hint:6,insid:[4,6],instal:1,instanc:6,instruct:6,int64:6,integ:[3,6],integr:3,intend:4,interfac:6,intermedi:4,interv:6,io:4,ipsc:3,its:[4,6],izhikevich:[3,6],jump:3,k:4,kei:[1,4],kera:[1,3,4,5,6],kernal:3,kernel:[1,3,4,5],kwarg:[3,4,6],lambda:6,larger:3,last:6,latter:4,layer:[1,4,5],lead:5,leaki:3,learn:[4,5],length:[3,4],lif:[1,3,6],like:[5,6],line:6,list:[3,4,6],load:6,locat:[],logic:[],lorenz:1,loss:[4,6],m:6,mae:6,mai:[3,5,6],main:[],mask:[3,4,6],match:[4,6],math:6,matric:4,matrix:[3,4,6],mean:[3,5,6],meet:6,membran:3,method:[3,4,6],metric:[4,6],mi:6,minimum:6,minval:6,model:[1,3],modifi:6,more:[3,4],multipl:6,must:[3,4,6],n:2,name:6,necessari:4,need:[4,6],network:[1,4,5,6],neural:[1,3,4],neuron:[3,4,6],next:[],nicola:[3,4],no_fb_esn_lay:6,nofeedbackesn:[3,6],nois:3,noise_param:3,noise_se:[],non:[3,6],none:[3,4,6],normal:6,note:[1,3,4,6],np:6,num_step:6,number:[3,4,6],numpi:[3,4,6],object:[3,4,6],one:[3,4,6],onli:[3,6],open:[],optim:[3,4],optimizedforcemodel:4,optimizedspikingnn:[3,6],option:3,order:[1,3],org:3,other:[4,6],otherwis:[],out:[3,4,6],outlin:[],output:[3,4,5,6],output_kernel:[3,6],output_kernel_train:3,output_s:[3,6],output_unit:6,overal:[],p:[4,6],p_gx:4,p_recurr:[3,6],p_task:4,packag:[1,4],paper:3,paramet:[3,4,6],parent:3,part:3,pass:[3,4,5,6],path:2,peak:[3,6],per:4,perform:[3,4,5,6],ph:4,pht:4,pip:2,posit:6,post:[3,6],potenti:3,pre:[3,6],predict:[3,4,6],present:[3,4],prev_a:6,prev_h:6,prev_output:6,previou:[],print:6,prior:[5,6],process:[3,6],project:1,properti:[3,6],pseudogradi:[1,4],pseudogradient_p:[4,6],pseudogradient_p_gx:[4,6],pseudogradient_wo:[4,6],pseudogradient_wr:[4,6],pseudogradient_wr_task:4,python:[1,2,3],q:3,quick:1,r:3,random:[1,3,6],randomli:3,randomnorm:6,rate:[3,4,6],read_valu:6,reciproc:3,recurr:[1,3,4,5,6],recurrent_kernel:[3,6],recurrent_kernel_train:3,recurrent_nontrainable_boolean_mask:[3,6],recurrent_term:6,recurrent_units1:6,recurrent_units2:6,reduc:1,refer:6,refractori:3,regular:6,relev:[4,6],remov:[],repo:2,repres:3,reproduc:6,requir:[3,4,6],research:[],reset:[3,4],reset_st:[4,6],reson:3,resonance_param:3,respect:[3,4,6],rest:3,result:[5,6],return_sequ:[4,6],rheobas:3,rise:3,rnn:[3,5,6],row:6,rtype:[],rule:[4,5,6],s:[3,4,6],same:[3,4],satur:5,save:6,scale:3,scheme:6,second:6,see:[3,4,6],seed:[3,6],seed_gen:6,seen:5,self:[3,4,6],sequenc:4,serv:4,set:[3,5,6],shape:[3,4,5,6],shift:5,should:[4,6],shuffl:4,signal:4,simpli:[],sinusoid:1,size:[1,3,4,5],slide:[],specifi:3,speed:3,spike:[1,4],spikingnn:[3,6],spikingnnmodel:[4,6],stack:[4,6],stall:5,standard:3,start:1,state:[1,4,5],state_s:[3,6],stddev:6,step:[3,4,6],steps_per_epoch:4,storag:[3,6],str:[3,4],strength:3,string:3,structur:3,structural_connect:3,style:[4,6],sub:[3,4,6],subclass:[3,4,6],sum:[1,4],support:[1,3,4,5],sussillo:4,svg:[],synapt:3,t_step:[3,6],tanh:[3,5,6],target:[3,4,6],target_output_kernel_train:[4,6],task:4,tau_decai:3,tau_mem:3,tau_ref:3,tau_ris:3,tau_syn:3,tension:[2,6],tensor:[3,4,6],tensorflow:[1,3,4,6],tensorshap:3,tf:[3,6],th:6,therefor:[3,4],theta:[1,3,6],thi:[1,3,4,6],third:6,those:6,three:[],threshold:3,through:5,thsi:[],thu:[3,6],time:[3,4,6],timestep:[4,5,6],trace:[3,6],track:[3,6],tradit:5,train:[1,3,4,5,6],train_step:[1,4],trainabl:[3,4,5,6],trainable_var:[4,6],trainable_vari:[4,6],trainable_vars_output_kernel:4,trainable_vars_p_gx:4,trainable_vars_p_output:4,trainable_vars_recurrent_kernel:4,treat:5,tree:[],truth:4,tupl:[3,4],two:[3,6],type:[],typic:[3,4,6],u:[3,6],under:1,uniform:6,unit:[3,4,6],unus:[3,4,6],updat:[1,3,4,5],update_firing_r:3,update_kernel_condit:6,update_output_kernel:[4,6],update_recurrent_kernel:[4,6],update_st:6,update_voltag:[3,6],us:[2,3,4,6],usag:1,user:[],usual:[],v:[3,6],v_mask:3,v_peak:[3,6],v_reset:3,v_rest:3,v_thre:3,valid:4,validation_batch_s:4,validation_step:4,valu:3,variabl:[3,4,6],varianc:6,vector:6,venv:2,verbos:4,version:4,via:[4,5,6],virtual:2,voltag:[3,6],wa:5,warn:6,weight:[3,4,5,6],well:[],when:[3,4,6],where:[3,6],whether:[3,4],which:6,white:3,width:3,within:3,without:[3,4,6],wo:4,word:4,wr:4,wr_target:4,wr_task:4,wrapper:4,write:6,www:3,x:[3,4,5,6],y:[4,6],z:[4,6],zebrafish:1,zero:[3,6],zhenruiliao:2},titles:["Examples","Welcome to TENSION\u2019s documentation!","Installation","FORCE Layers","FORCE Model","Notes","Usage"],titleterms:{"class":6,"new":6,NO:0,access:6,advanc:[],attractor:0,attribut:6,base:[3,4],build:6,callback:6,compat:6,content:1,creat:6,custom:6,data:0,document:1,echo:[0,3],exampl:0,forc:[0,3,4],force_layer_cal:6,forcelay:6,forcemodel:6,from_weight:6,full:0,go:0,gpu:6,inherit:4,initi:6,instal:2,kei:6,kernel:6,layer:[3,6],lif:0,lorenz:0,model:[4,6],network:[0,3],neural:0,note:5,pseudogradi:6,quick:6,random:0,s:1,sinusoid:0,size:6,spike:[0,3,6],start:[0,6],state:[0,3,6],sum:0,support:6,tension:1,theta:0,train:0,train_step:6,updat:6,usag:6,welcom:1,zebrafish:0}})