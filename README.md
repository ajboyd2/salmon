# _SALMON_

Salmon is a package for symbolic algebra of linear regression and modeling. The goal for this package is to ease the process of model building for linear regression by separating the model (with all its interactions and variables) from the data being used to fit it. 

If you would like to use Salmon, you can install it by first cloning the repository, navigating to the repository directory, and execute the following commands (inside a virtual environment if you prefer):

```
> python setup.py build
> python setup.py install
```

From there, the package should be installed into your Python environment and can be accessed by importing the package name `salmon`.

For the purpose of this documentation we will be using the Harris Bank dataset for our examples.

Using salmon can be defined in three stages:

1. Defining the model
2. Fitting the model
3. Using the model

As such, the documentation will be broken up into these three parts.

```python
# Setup
import pandas as pd
from salmon import *
%matplotlib inline
data = pd.read_csv("./data/harris.csv")
data = data[data['Educ'] != 10].reset_index(drop=True) # Remove single outlier
```

# Model Definition

A model is defined by a quatitative or categorical variables existing either stand alone, within interactions, within linear combinations, or a mix of the latter two.

### Variable Types

A variable in this package represts the variables you would commonly see in a definition of a regression model like so:
$$f(\tt{Var}) = \beta_0 + \beta_1 \tt{Var_1} + \beta_2 \tt{Var_2} + \beta_3 \tt{Var_2}^2 + \beta_4 \tt{Var_1}*{Var_2} + \beta_5 \tt{Var_1} * {Var_2}^2$$
where $\tt{Var_i}$ represents either a quantitative or categorical variable. 

Salmon represents these symbolically. When defining the variables, there are three options to pick from:


```python
# Variables known to be quantitative.
quant_var = Q("Bsal")
# Variables known to be categorical.
cat_var = C("Sex")
# Variables of unknown type. These will be interpreted when fitting as either categorical or quantitative.
interp_var = Var("Educ")
```

The string passed into the variables are the column names to extract from a pandas `DataFrame` when fitting on a set of data. So for instance, if we defined a model with `Q("Bsal")` then the model would extract the `Bsal` column to work with from the data passed in.

#### Quantitative Variables
Common transformations of quantitative variables are also supported. For example:


```python
bsal_squared = Q("Bsal") ** 2
bsal_shifted = Q("Bsal") + 150
bsal_logged = Log(Q("Bsal"))
```

#### Categorical Variables

When defining categorical variables it is possible to set ahead of time the possible levels/factors to fit with, as well as the encoding method for use. For instance, if we wanted to treat the `Educ` column in our example dataset as a categorical variable, and we knew the possible levels of education were either 8, 10, 12, 15, or 16, then we could define our variable as follows:


```python
educ_var_v1 = C("Educ", method = 'one-hot', levels = [8, 10, 12, 15, 16])
educ_var_v2 = C("Educ", method = 'one-hot', levels = [8, 10, 12])
```

The first variable defined set the order of the levels to interpreted as. This would matter with encoding methods such as ordinal encoding (note: currently not supported, only one-hot encoding is supported at this time). In our case with the one-hot encoding method used, our ordering designated the '8' level to be dropped to avoid multi-colinearity.

The second variable defined still designated the '8' level to be dropped; however, it also designates that any levels found in the data that are not either '8', '10', or '12' will be binned into an 'other' category.

By default, categorical variables will use a one-hot encoding method and will dynamically extract the possible levels of a variable upon fitting. The levels will be ordered by sorting and the smallest (according to Python's `sorted` function) level will be dropped. 

## Combinations

Many regression applications require multiple variables within the model. This is achieved in salmon by simply adding together several variables. For instance, suppose we wanted to represent:
$$\tt{Sex} + \tt{Bsal} + \tt{Bsal}^2$$
This would be achieved like so:


```python
combo = C("Sex") + Q("Bsal") + Q("Bsal") ** 2
```

Should you want to define an full polynomial sequence you can use the following command as well:


```python
combo = C("Sex") + Q("Bsal", 4) # Expands to 'Sex + Bsal + Bsal^2 + Bsal^3 + Bsal^4'
```

## Interactions

It is common to want to model interaction effects between variables. Salmon supports this symbolically using the `*` operator. Any combination of variable type is supported.

For example, let's model this interaction:
$$\tt{Sex * Bsal}$$


```python
interaction = C("Sex") * Q("Bsal")
```

Here is how we would model a more complicated linear combination of variables and interactions like $$\tt{Sex} + \tt{Bsal} + \tt{Bsal}^2 + \tt{Sex * Bsal} + \tt{Sex}*{Bsal}^2$$



```python
complicated_combo = C("Sex") + Q("Bsal") + Q("Bsal")**2 + C("Sex")*Q("Bsal") + C("Sex")*Q("Bsal")**2
```

Salmon also supports distribution of singular terms into combinations. The above expression could be represented more succinctly as such:


```python
# Equivalent to C("Sex") + Q("Bsal") + Q("Bsal")**2 + (Q("Bsal") + Q("Bsal")**2) * C("Sex") 
complicated_combo = C("Sex") + Poly("Bsal", 2) + Poly("Bsal", 2) * C("Sex")
```

## Representing the Model

Now that we understand how to form expressions of variables, we can now represent our models. `LinearModels` are always defined of the form:

`model = LinearModel(explanatory_expression, response_expression)`

The `explanatory_expression` is allowed to be a single term, an interaction, or a combination of the other two. The `response_expression` is allowed to be either a single term or an interaction. Categorical variables are allowed within the `response_expression` so long after encoding the resultant expansion is represented by only one column.

# Fitting the Model

For an example, let us fit this model:

$$\widehat{Sal77}(Sex, Bsal) = \beta_0 + \beta_1 \tt{Sex} + \beta_2 \tt{Bsal} + \beta_3 \tt{Bsal}^2 + \beta_4 \tt{Sex * Bsal} + \beta_5 \tt{Sex}*{Bsal}^2$$

First we must define our model:


```python
explanatory = C("Sex") + Poly("Bsal", 2) + C("Sex") * Poly("Bsal", 2)
response = Q("Sal77")
model = LinearModel(explanatory, response)
```

Note how we did not need a term for the $\beta_0$ (the intercept). This is because it is not a part of our explantory expression of variables, but rather inherent in the model definition. Should we have wanted to define our model without an intercept, we would define it as `LinearModel(explanatory, response, intercept = False)`

Now that we have our model defined, we must fit the data to it for it to compute all $\beta_i$ values. We do this like so:


```python
model.fit(data)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficients</th>
      <th>SE</th>
      <th>t</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>15244.585697</td>
      <td>15004.866623</td>
      <td>1.015976</td>
      <td>0.312491</td>
    </tr>
    <tr>
      <th>Sex::Male</th>
      <td>-15746.353511</td>
      <td>20175.689005</td>
      <td>-0.780462</td>
      <td>0.437262</td>
    </tr>
    <tr>
      <th>Bsal</th>
      <td>-2.333748</td>
      <td>5.847857</td>
      <td>-0.399078</td>
      <td>0.690825</td>
    </tr>
    <tr>
      <th>(Bsal)^2</th>
      <td>0.000242</td>
      <td>0.000566</td>
      <td>0.427475</td>
      <td>0.670102</td>
    </tr>
    <tr>
      <th>{Bsal}{Sex::Male}</th>
      <td>5.472859</td>
      <td>7.291904</td>
      <td>0.750539</td>
      <td>0.454979</td>
    </tr>
    <tr>
      <th>{(Bsal)^2}{Sex::Male}</th>
      <td>-0.000423</td>
      <td>0.000666</td>
      <td>-0.636137</td>
      <td>0.526377</td>
    </tr>
  </tbody>
</table>
</div>



Notice how we did not designate datasets separately for the explantory and response. The model assumed all variables used when defined will be found within the one `DataFrame` passed in as an argument. Also notice how we did not have to transform our original dataset to include the transformations and interactions. This was all done interally at runtime while fitting the data to the model.

# Using the Model

Now that our model is fit, we can do a variety of things with it.

First off, the most common use of a model would be to make predictions with new data. This would be done like so:


```python
model.predict(data)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted Sal77</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10714.633449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12079.759519</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12488.612620</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13031.467692</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12527.514782</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12527.514782</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11163.403112</td>
    </tr>
    <tr>
      <th>12</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>13</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>14</th>
      <td>9640.923816</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9621.847633</td>
    </tr>
    <tr>
      <th>16</th>
      <td>9673.291727</td>
    </tr>
    <tr>
      <th>17</th>
      <td>9673.291727</td>
    </tr>
    <tr>
      <th>18</th>
      <td>9621.847633</td>
    </tr>
    <tr>
      <th>19</th>
      <td>9621.847633</td>
    </tr>
    <tr>
      <th>20</th>
      <td>9703.587918</td>
    </tr>
    <tr>
      <th>21</th>
      <td>9740.858177</td>
    </tr>
    <tr>
      <th>22</th>
      <td>9703.587918</td>
    </tr>
    <tr>
      <th>23</th>
      <td>9809.839940</td>
    </tr>
    <tr>
      <th>24</th>
      <td>9826.146597</td>
    </tr>
    <tr>
      <th>25</th>
      <td>9621.847633</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10031.820474</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9660.758907</td>
    </tr>
    <tr>
      <th>28</th>
      <td>9640.923816</td>
    </tr>
    <tr>
      <th>29</th>
      <td>9668.368680</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>62</th>
      <td>12319.952051</td>
    </tr>
    <tr>
      <th>63</th>
      <td>11501.485049</td>
    </tr>
    <tr>
      <th>64</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>65</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>66</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>67</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>68</th>
      <td>10131.682604</td>
    </tr>
    <tr>
      <th>69</th>
      <td>10944.891645</td>
    </tr>
    <tr>
      <th>70</th>
      <td>12319.952051</td>
    </tr>
    <tr>
      <th>71</th>
      <td>11163.403112</td>
    </tr>
    <tr>
      <th>72</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>73</th>
      <td>11163.403112</td>
    </tr>
    <tr>
      <th>74</th>
      <td>11806.937185</td>
    </tr>
    <tr>
      <th>75</th>
      <td>9809.839940</td>
    </tr>
    <tr>
      <th>76</th>
      <td>9703.587918</td>
    </tr>
    <tr>
      <th>77</th>
      <td>9656.492266</td>
    </tr>
    <tr>
      <th>78</th>
      <td>10153.107740</td>
    </tr>
    <tr>
      <th>79</th>
      <td>9959.679880</td>
    </tr>
    <tr>
      <th>80</th>
      <td>9640.923816</td>
    </tr>
    <tr>
      <th>81</th>
      <td>9621.847633</td>
    </tr>
    <tr>
      <th>82</th>
      <td>9640.923816</td>
    </tr>
    <tr>
      <th>83</th>
      <td>9809.839940</td>
    </tr>
    <tr>
      <th>84</th>
      <td>9703.587918</td>
    </tr>
    <tr>
      <th>85</th>
      <td>9640.923816</td>
    </tr>
    <tr>
      <th>86</th>
      <td>9621.847633</td>
    </tr>
    <tr>
      <th>87</th>
      <td>9959.679880</td>
    </tr>
    <tr>
      <th>88</th>
      <td>9668.368680</td>
    </tr>
    <tr>
      <th>89</th>
      <td>9762.108581</td>
    </tr>
    <tr>
      <th>90</th>
      <td>9631.324124</td>
    </tr>
    <tr>
      <th>91</th>
      <td>9660.758907</td>
    </tr>
  </tbody>
</table>
<p>92 rows × 1 columns</p>
</div>



The only restriction is that the new `DataFrame` being passed in must have enough columns with the necessary names used to define the model originally.

## Plotting

Should the model's definition fall under certain categories, plotting the original training data against the linear fit is available as well. As of right now, plotting supports models with explantory expressions consisting of only categorical variables, or expressions consisting of only one quantitative variable and zero or more categorical variables. Some example plots would look as follows:


```python
ex_explanatory = C("Sex")
ex_response = Q("Sal77")
ex_model = LinearModel(ex_explanatory, ex_response)
ex_model.fit(data)
ex_model.plot()
```


![png](SALMON%20Documentation_files/SALMON%20Documentation_25_0.png)



```python
ex_explanatory = C("Sex") + C("Educ") + C("Sex") * C("Educ")
ex_response = Q("Sal77")
ex_model = LinearModel(ex_explanatory, ex_response)
ex_model.fit(data)
ex_model.plot()
```


![png](SALMON%20Documentation_files/SALMON%20Documentation_26_0.png)



```python
ex_explanatory = Q("Bsal")
ex_response = Q("Sal77")
ex_model = LinearModel(ex_explanatory, ex_response)
ex_model.fit(data)
ex_model.plot()
```


![png](SALMON%20Documentation_files/SALMON%20Documentation_27_0.png)



```python
ex_explanatory = Poly("Bsal", 2)
ex_response = Q("Sal77")
ex_model = LinearModel(ex_explanatory, ex_response)
ex_model.fit(data)
ex_model.plot()
```


![png](SALMON%20Documentation_files/SALMON%20Documentation_28_0.png)



```python
ex_explanatory = C("Sex") + Poly("Bsal", 2) + C("Sex") * Poly("Bsal", 2)
ex_response = Q("Sal77")
ex_model = LinearModel(ex_explanatory, ex_response)
ex_model.fit(data)
ex_model.plot()
```


![png](SALMON%20Documentation_files/SALMON%20Documentation_29_0.png)


## Diagnostics

There are two main diagnostic tools that salmon offers to allow you to evaluate the performance of your model and assure that no assumptions are broken: residual plots and partial regression plots. They can be accessed like so:


```python
model.residual_plots()
```


![png](SALMON%20Documentation_files/SALMON%20Documentation_31_0.png)



![png](SALMON%20Documentation_files/SALMON%20Documentation_31_1.png)



![png](SALMON%20Documentation_files/SALMON%20Documentation_31_2.png)



![png](SALMON%20Documentation_files/SALMON%20Documentation_31_3.png)



![png](SALMON%20Documentation_files/SALMON%20Documentation_31_4.png)



![png](SALMON%20Documentation_files/SALMON%20Documentation_31_5.png)





    [<matplotlib.collections.PathCollection at 0x2a232239d68>,
     <matplotlib.collections.PathCollection at 0x2a232686f60>,
     <matplotlib.collections.PathCollection at 0x2a232cc5d68>,
     <matplotlib.collections.PathCollection at 0x2a232d18e10>,
     <matplotlib.collections.PathCollection at 0x2a232d82c18>,
     <matplotlib.collections.PathCollection at 0x2a232dea780>]




```python
model.partial_plots()
```


![png](SALMON%20Documentation_files/SALMON%20Documentation_32_0.png)



![png](SALMON%20Documentation_files/SALMON%20Documentation_32_1.png)



![png](SALMON%20Documentation_files/SALMON%20Documentation_32_2.png)



![png](SALMON%20Documentation_files/SALMON%20Documentation_32_3.png)



![png](SALMON%20Documentation_files/SALMON%20Documentation_32_4.png)

