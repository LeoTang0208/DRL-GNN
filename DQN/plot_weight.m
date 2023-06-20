clear;
clc;

s = [1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12];
t = [2, 3, 4, 3, 8, 6, 5, 9, 6, 7, 13, 14, 8, 11, 10, 12, 11, 13, 12, 14, 13];

% PS D:\DRL-GNN\DQN> python evaluate_DQN.py -d ./Logs/expsample_DQN_agent_plr_4_Logs.txt -s 0 -e 0
% >>>>> lb
% [166, 169, 213, 163, 202, 234, 207, 202, 220, 154, 160, 143, 145, 170, 166, 134, 158, 165, 150, 144, 147]
% >>>>> sap
% [128, 113, 213, 100, 183, 200, 193, 240, 195, 135, 194, 147, 124, 216, 154, 102, 156, 112, 105, 111, 64]
% >>>>> dqn
% [158, 151, 221, 125, 200, 173, 182, 243, 147, 147, 229, 139, 144, 220, 156, 127, 170, 161, 136, 139, 145]
 
% PS D:\DRL-GNN\DQN> python evaluate_DQN.py -d ./Logs/expsample_DQN_agent_orig_4_Logs.txt -s 0 -e 0
% >>>>> lb
% [161, 142, 189, 168, 192, 208, 168, 155, 191, 159, 164, 173, 133, 189, 136, 124, 138, 134, 152, 151, 121]
% >>>>> sap
% [143, 97, 189, 120, 210, 203, 181, 190, 170, 137, 183, 147, 120, 213, 150, 91, 144, 101, 96, 120, 59]
% >>>>> dqn
% [161, 156, 173, 158, 214, 205, 161, 192, 177, 175, 201, 173, 180, 208, 144, 129, 151, 144, 136, 150, 124]

% PS D:\DRL-GNN\DQN> python evaluate_DQN.py -d ./Logs/expsample_DQN_agent_orig_4_Logs.txt -s 0 -e 0
% >>>>> lb
% [120, 121, 162, 122, 155, 155, 168, 149, 175, 96, 146, 110, 93, 131, 121, 51, 117, 121, 75, 106, 55]
% >>>>> sap
% [96, 68, 130, 91, 162, 131, 132, 159, 125, 85, 117, 116, 83, 157, 121, 43, 116, 77, 67, 109, 26]
% >>>>> dqn
% [104, 83, 141, 87, 134, 246, 167, 134, 243, 81, 146, 88, 52, 100, 78, 29, 70, 78, 42, 75, 44]

drl_w = [104, 83, 141, 87, 134, 246, 167, 134, 243, 81, 146, 88, 52, 100, 78, 29, 70, 78, 42, 75, 44];
sap_w = [96, 68, 130, 91, 162, 131, 132, 159, 125, 85, 117, 116, 83, 157, 121, 43, 116, 77, 67, 109, 26];
lb_w = [120, 121, 162, 122, 155, 155, 168, 149, 175, 96, 146, 110, 93, 131, 121, 51, 117, 121, 75, 106, 55];

drl_w = mapminmax(drl_w, 1, 5);
sap_w = mapminmax(sap_w, 1, 5);
lb_w = mapminmax(lb_w, 1, 5);

G_drl = graph(s, t, drl_w);
f1 = figure;
h1 = plot(G_drl);
h1.LineWidth = drl_w;
h1.EdgeCData = drl_w;
set(gca, 'xtick', [], 'xticklabel', [])
set(gca, 'ytick', [], 'yticklabel', [])
colormap(jet);
c1 = colorbar;
c1.Ticks = [];
title("Deep Reinforcement Learning", "fontsize", 12);

G_sap = graph(s, t, sap_w);
f2 = figure;
h2 = plot(G_sap);
h2.LineWidth = sap_w;
h2.EdgeCData = sap_w;
set(gca, 'xtick', [], 'xticklabel', [])
set(gca, 'ytick', [], 'yticklabel', [])
colormap(jet);
c2 = colorbar;
c2.Ticks = [];
title("Shortest Available Path", "fontsize", 12);

G_lb = graph(s, t, lb_w);
f3 = figure;
h3 = plot(G_lb);
h3.LineWidth = lb_w;
h3.EdgeCData = lb_w;
set(gca, 'xtick', [], 'xticklabel', [])
set(gca, 'ytick', [], 'yticklabel', [])
colormap(jet);
c3 = colorbar;
c3.Ticks = [];
title("Load Balancing", "fontsize", 12);