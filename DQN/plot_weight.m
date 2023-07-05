clear;
clc;

s = [1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12];
t = [2, 3, 4, 3, 8, 6, 5, 9, 6, 7, 13, 14, 8, 11, 10, 12, 11, 13, 12, 14, 13];

e = [0.044, 0.024, 0.003, 0.023, 0.079, 0.094, 0.090, 0.060, 0.010, 0.062, 0.002, 0.081, 0.060, 0.011, 0.034, 0.058, 0.029, 0.017, 0.097, 0.066, 0.021];
e = e * 100;


drl_w = [102, 110, 188, 113, 182, 220, 200, 224, 186, 147, 209, 136, 125, 227, 169, 75, 167, 138, 102, 119, 56];
sap_w = [130, 157, 202, 124, 198, 195, 186, 241, 187, 184, 214, 180, 184, 218, 171, 141, 160, 148, 138, 155, 141];
lb_w = [186, 165, 206, 167, 229, 242, 204, 186, 216, 171, 180, 165, 168, 184, 168, 138, 162, 168, 141, 155, 151];

drl_w = mapminmax(drl_w, 1, 5);
sap_w = mapminmax(sap_w, 1, 5);
lb_w = mapminmax(lb_w, 1, 5);

G_drl = graph(s, t, drl_w);
f1 = figure;
h1 = plot(G_drl, 'EdgeLabel', e);
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
h2 = plot(G_sap, 'EdgeLabel', e);
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
h3 = plot(G_lb, 'EdgeLabel', e);
h3.LineWidth = lb_w;
h3.EdgeCData = lb_w;
set(gca, 'xtick', [], 'xticklabel', [])
set(gca, 'ytick', [], 'yticklabel', [])
colormap(jet);
c3 = colorbar;
c3.Ticks = [];
title("Load Balancing", "fontsize", 12);