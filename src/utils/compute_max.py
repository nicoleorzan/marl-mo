import numpy as np
import matplotlib.pyplot as plt


def value_function(p1, f, beta, c1):
    return f/2.*(2*c1*p1)**beta + c1*(1-p1)

def p_value(f, beta, c1):
    return 1./(c1*f)*(2./(beta*f))**(1./(beta-1.))

def der_seconda(f, beta, c1):
    return beta*(f*c1)**2./4.*(beta-1.)*(2/(beta*f))**((beta-2.)/(beta-1.))



f = 3.
c1 = 4.

#print("f=", f, "beta=", beta, "c1=", c1)

#print("p value=", p_value(f, beta, c1))
#print("der seconda=", der_seconda(f, beta, c1))


betas = np.linspace(0,5,100)
p_values = [p_value(f, betas[i], c1) for i in range(len(betas))]
value_func = [value_function(p_values[i], f, betas[i], c1) for i in range(len(betas))]
plt.plot(betas, value_func)
#plt.plot(betas, p_value(f, betas, c1))
plt.ylim(0,10)
plt.show()