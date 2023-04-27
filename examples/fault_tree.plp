% Example of Fault Tree Analysis using belief functions taken from:
% Uncertain Evidence and Artificial Analysis.
% A.P. Dempster and A. Kong.
% Journal of Statistical Planning and Inference, vol. 20, pp. 355-368, 1988

% fault tree (see Fig. 1 of the article)
m :- f. m :- g.
f :- e. f :- c.
e :- d. e :- x(4).
c :- x(4). c :- x(5).
d :- a, b.
a :- x(1). a :- x(3).
b :- x(3). b :- x(2).

% multivalue mappings of variables
x(V) :- p(V,1).
y(V) :- p(V,0).
x(V); y(V) :- p(V,all).

g :- pG(1).
ng :- pG(0).
g; ng :- pG(all).

% m-functions (from Table 1 of the article)
0.05::p(1,1);  0.9::p(1,0);   0.05::p(1,all).
0.05::p(2,1);  0.9::p(2,0);   0.05::p(2,all).
0.005::p(3,1); 0.99::p(3,0);  0.005::p(3,all).
0.01::p(4,1);  0.985::p(4,0); 0.005::p(4,all).
0.002::p(5,1); 0.995::p(5,0); 0.003::p(5,all).
0.001::pG(1);  0.99::pG(0);   0.009::pG(all).

% inference
#query(x(1)).
#query(x(2)).
#query(x(3)).
#query(a).
#query(b).
#query(c).
#query(d).
#query(e).
#query(f).
#query(g).
#query(m).

% conditional inferences
#query(g | m).
#query(f | m).
#query(a | m).
#query(x(1) | m).

