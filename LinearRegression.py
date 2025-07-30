import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1,11)
y = [20, 21, 24, 29, 36, 39, 40, 45, 47, 50]
y = np.array(y)
plt.scatter(x,y, marker = "*")
plt.show()

#find line of best fit
#m = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
#c = mean(y) – m * mean(x)
#Simple linear regression - one feature and one target

meanx = np.mean(x)
meany = np.mean(y)
m = np.sum((x - meanx)*(y-meany)) / np.sum(((x - meanx)**2))
c = meany - m*meanx
print("Gradient = ", m, ", Y intercept = ", c)

#Prediction
predy = m*x + c
print(predy)
plt.scatter(x,y, marker = "*")
plt.plot(x, predy, "g-o")
plt.show()

#evaluating the model
#RMSE - Root Mean Squared Error sqrt( sum( (p – yi)^2 )/n )
rmse = np.sqrt(np.mean((predy - y)**2))
print(rmse)