# Code to demonstrate ChiSquare in Python
from scipy.stats import chi2_contingency
print('Solving the First Problem...')
# defining the data table for the first problem 
data = [[100,70,30], [140,60,20]]
stat, p, dof, expected = chi2_contingency(data)

# interpret p-value
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (H0 holds true)')

# Output: p value is 0.0142; Dependent (Reject H0)
# alpha of 0.05 > 0.0142

print('Solving the Second Problem...')
# defining the data table for the Second problem 
data = [[11,5,1],[8,6,8],[3,10,12]]
stat, p, dof, expected = chi2_contingency(data)

# interpret p-value
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (H0 holds true)')

# Output: p value is 0.0059; Dependent (Reject H0)
# alpha of 0.05 > 0.0059

print('One example where H0 is accepted')
data = [[207, 282, 241], [234, 242, 232]]
stat, p, dof, expected = chi2_contingency(data)
  
# interpret p-value
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')

# Output: p value is 0.103; Independent (H0 holds true)
# alpha of 0.05 < 0.103 -> Accept H0, that is, the variables do not have a significant relation.