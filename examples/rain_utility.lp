% This program encodes the illustrative example of decision making with ProbLog described in
%
%     G. Van den Broeck, I. Thon, M. van Otterlo and L. De Raedt. DTProbLog: A decision-theoretic
%     probabilistic Prolog. Proceedings of the twenty-fourth AAAI conference on artificial
%     intelligence, pp. 1217 - 1222, AAAI Press, 2010.
%
% Decisions are encoded as vacuous probabilistic facts.
% Utilities are converted into probabilities by Coopers linear transformation
%   (U-min U)/(max U - min U),
% where max/min U are the maximum/minimum utilities (60 and -40 for this problem). An additive
% combined utility function is encoded by means of auxiliary atoms.

% We want to decided whether we should bring an umbrella, and whether we should bring a raincoat.
% These are independent decisions.
[0,1]::umbrella.
[0,1]::raincoat.

% There is a 30% of chance of raining and 50% of being windy
0.3::rain.
0.5::wind.

% When it is raining and windy, the umbrella breaks.
broken_umbrella :- umbrella, rain, wind.
% We remain dry if it is not raining or if we wear a raincoat or an unbroken umbrella.
dry :- rain, raincoat.
dry :- rain, umbrella, not broken_umbrella.
dry :- not rain.

% Bringing a broken umbrella costs 40 utilities:
%   U(broken_umbrella) = -40
% We need to convert both the utility of a broken umbrella...
%   P(util(broken_umbrella)) = [U(broken_umbrella)-(-40)]/[60-(-40)]=0
% (since the rule has probability zero, there is no benefit in adding it to the program)
%   0.0::util(broken_umbrella) :- broken_umbrella.
% ...and the utility of not having a broken umbrella (which is zero before the linear
% transformation)
%   P(util(broken_umbrella)) = [U(not broken_umbrella)-(-40)]/[60-(-40)]=0.4
0.4::util(broken_umbrella) :- not broken_umbrella.
% Bringing a raincoat costs 20 utils.
%   U(raincoat) = -20
0.2::util(raincoat) :- raincoat.
% Not bringing it costs nothing.
0.4::util(raincoat) :- not raincoat.
% Bringing an umbrella costs 2 utils.
%   U(umbrella) = -2
0.38::util(umbrella) :- umbrella.
% Not bringing it costs zero.
0.4::util(umbrella) :- not umbrella.
% We want to stay dry.
%   U(dry) = 60
% A rule with probability 1 is simply a deterministic rule.
util(dry) :- dry.
% Not staying dry has utility 0.
0.4::util(dry) :- not dry.

% We encode the sum of utilities of broken_umbrella and rain coat as follows.

% P(add(broken_umbrella, raincoat)) = 0.5*[P(util(broken_umbrella))+P(util(raincoat))]
0.5::add(broken_umbrella, raincoat) :- util(broken_umbrella), not util(raincoat).
0.5::add(broken_umbrella, raincoat) :- not util(broken_umbrella), util(raincoat).
add(broken_umbrella, raincoat) :- util(broken_umbrella), util(raincoat).

% P(add(umbrella, dry)) = 0.5*[P(util(umbrella))+P(util(dry))]
0.5::add(umbrella, dry) :- util(umbrella), not util(dry).
0.5::add(umbrella, dry) :- not util(umbrella), util(dry).
add(umbrella, dry) :- util(umbrella), util(dry).

% We add the intermediate sums:

% P(utility) = 0.5*[P(add(broken_umbrella, raincoat))+P(add(umbrella, dry))]
0.5::utility :- add(broken_umbrella, raincoat), not add(umbrella, dry).
0.5::utility :- not add(broken_umbrella, raincoat), add(umbrella, dry).
utility :- add(broken_umbrella, raincoat), add(umbrella, dry).

% The result is computed by
% P(utility) = (1/4)*(1/100)*[U(...) + 4*40] = 0.5075
% In particular, max Exp. Utility = P(utility)*4*100-4*40 = 43
#query(utility).

