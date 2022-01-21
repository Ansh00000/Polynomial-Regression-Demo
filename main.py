from res.imports import *

# Fetching data
x = dc.Data().get_data().iloc[:, 1: -1]
y = dc.Data().get_data().iloc[:, -1]

# Applying linear regression
L_regressor = LinearRegression()
L_regressor.fit(x, y)

# applying polynomial regression
p_regressor = PolynomialFeatures(degree=4)
x_poly = p_regressor.fit_transform(x)
L_regressor2 = LinearRegression()
L_regressor2.fit(x_poly, y)

# Visualizing the linear regression curve
plt.scatter(x, y, color='red')
plt.plot(x, L_regressor.predict(x), color='blue')
plt.title('Level VS Salary')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()

# Visualizing the Polynomial regression curve
plt.scatter(x, y, color='red')
plt.plot(x, L_regressor2.predict(p_regressor.fit_transform(x)), color='blue')
plt.title('Level VS Salary')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()

# predicting salary based on Linear regression
print(L_regressor.predict([[6.5]]))

# predicting salary based on Polynomial regression
print(L_regressor2.predict(p_regressor.fit_transform([[6.5]])))
