clear;
clc;

s = [1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12];
t = [2, 3, 4, 3, 8, 6, 5, 9, 6, 7, 13, 14, 8, 11, 10, 12, 11, 13, 12, 14, 13];

drl_w = [161, 156, 173, 158, 214, 205, 161, 192, 177, 175, 201, 173, 180, 208, 144, 129, 151, 144, 136, 150, 124];
sap_w = [143, 97, 189, 120, 210, 203, 181, 190, 170, 137, 183, 147, 120, 213, 150, 91, 144, 101, 96, 120, 59];
lb_w = [161, 142, 189, 168, 192, 208, 168, 155, 191, 159, 164, 173, 133, 189, 136, 124, 138, 134, 152, 151, 121];

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