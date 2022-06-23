% Popular Asia-Tuberculosis Bayesian Network.

0.01::trip. 0.5::smoking.
0.05::tuberculosis :- trip.
0.01::tuberculosis :- not trip.
0.1::cancer :- smoking.
0.01::cancer :- not smoking.
or :- tuberculosis. or :- cancer.
0.98::test :- or.
0.05::test :- not or.

#query(trip)
#query(tuberculosis | trip)
#query(cancer | smoking)
#query(test | or)
#query(smoking)
#query(tuberculosis | not trip)
#query(cancer | not smoking)
#query(test | not or)
