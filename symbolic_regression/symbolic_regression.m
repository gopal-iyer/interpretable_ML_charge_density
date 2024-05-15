function gp = gpdemo6_config(gp)
addpath(genpath('<path_to_gptips2>'))
%run control
gp.runcontrol.pop_size = 2500; %population of 250 models                    
gp.runcontrol.num_gen = 40;
gp.runcontrol.runs = 10; %perform 2 runs that are merged at the end
gp.runcontrol.timeout = 360; %each run terminates after 60 seconds
gp.runcontrol.parallel.auto = true; %enable Parallel Computing if installed

%selection
gp.selection.tournament.size = 20; 
gp.selection.tournament.p_pareto = 0.3; %encourages less complex models 
gp.selection.elite_fraction = 0.3; % approx. 1/3 models copied to next gen

%genes
gp.genes.max_genes = 10;  

% LOAD DATA FILES
x1=load('../Amat.txt');
y1=load('../bvec.txt');

n1=randperm(length(y1), 50000);


% TEST
gp.userdata.ytest = y1(n1);
gp.userdata.xtest = x1(n1,:);

% TRAIN
y1(n1)=[];
x1(n1,:)=[];
gp.userdata.ytrain = y1;
gp.userdata.xtrain = x1;

%function nodes
gp.nodes.functions.name = {'times','minus','plus','rdivide','square',...
			   'exp','mult3','add3','sqrt','cube','power','negexp','neg'};
