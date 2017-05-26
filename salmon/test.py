import unittest
from Expression import *
from Model import *
import pandas as pd

def floatComparison(a, b, eps = 0.0001):
    return(a - eps < b and b < a + eps)

# Expressions
class TestVarMethods(unittest.TestCase):
    
    def test_str(self):
        self.assertEqual(str(Var("Test Case")), "Test Case")
        
    def test_mul(self):
        self.assertEqual(str(Var("Var1") * Var("Var2")), "{Var1}{Var2}")
        self.assertEqual(str(Var("Test") * Poly("Single", 2)), "{Test}{Single} + {Test}{Single^2}")
        with self.assertRaises(Exception):
            Var("Test2") * 4
        
    def test_imul(self):
        v = Var("Test")
        v *= Var("Other")
        self.assertEqual(str(v), "{Test}{Other}")
        v = Var("Test")
        v *= Poly("Single", 2)
        self.assertEqual(str(v), "{Test}{Single} + {Test}{Single^2}")
        with self.assertRaises(Exception):
            v *= 4
                
    def test_flatten(self):
        flattened = Var("Test").flatten()
        self.assertEqual(len(flattened), 1)
        self.assertEqual(str(flattened[0]), "Test")
                
    def test_copy(self):
        orig = Var("A")
        copy = orig.copy()
        self.assertFalse(orig == copy)
        self.assertEqual(str(orig), str(copy))
                
    def test_interpret(self):
        data = pd.DataFrame({"A" : [1,2,3], "B" : ["hello", "test", "goodbye"]})
        var1 = Var("A")
        var2 = Var("B")
        self.assertTrue(isinstance(var1.interpret(data), Quantitative))
        self.assertTrue(isinstance(var2.interpret(data), Categorical))
                
class TestQuantitativeMethods(unittest.TestCase):
    
    def test_str(self):
        var = Quantitative("A", transformation = 2, coefficient = 4, shift = 3)
        self.assertEqual(str(var), "4*(A+3)^2")
        var = Quantitative("A", transformation = "sin", coefficient = 4, shift = 3)
        self.assertEqual(str(var), "4*sin(A+3)")
        
    def test_add(self):
        var1 = Quantitative("A")
        self.assertEqual(str(var1 + 3), "A+3")
        self.assertEqual(str(var1 + var1), "A + A")
        
    def test_iadd(self):
        var = Quantitative("A")
        var += 3
        self.assertEqual(str(var), "A+3")
        var += -3
        var += var
        self.assertEqual(str(var), "A + A")
                
    def test_mul(self):
        var = Quantitative("A")
        self.assertEqual(str(var * 3), "3*A")
        self.assertEqual(str(var * var), "A^2")
        self.assertEqual(str(var * Quantitative("B")), "{A}{B}")
                
    def test_imul(self):
        var1 = Quantitative("A")
        var1 *= 3
        self.assertEqual(str(var1), "3*A")
        var2 = Quantitative("A")
        var2 *= var2
        self.assertEqual(str(var2), "A^2")
        var3 = Quantitative("A")
        var3 *= Quantitative("B")
        self.assertEqual(str(var3), "{A}{B}")
        
    def test_pow(self):
        var = Quantitative("A")
        self.assertEqual(str(var ** 4), "A^4")
        with self.assertRaises(Exception):
            var ** "bad arg"
        with self.assertRaises(Exception):
            var = Log(var)
            var ** 43
                
    def test_ipow(self):
        var = Quantitative("A")
        var **= 3
        self.assertEqual(str(var), "A^3")
        with self.assertRaises(Exception):
            var *= "bad arg"
                
    def test_transform(self):
        var = Quantitative("A")
        var.transform("sin")
        self.assertEqual(str(var), "sin(A)")
        with self.assertRaises(Exception):
            var = Quantitative("A")
            var = var ** 3
            var.transform("bad call")

    def test_copy(self):
        orig = Quantitative("A")
        copy = orig.copy()
        self.assertFalse(orig == copy)
        self.assertEqual(str(orig), str(copy))

    def test_interpret(self):
        var = Quantitative("A")
        self.assertEqual(var.interpret(None), var)

class TestCategoricalMethods(unittest.TestCase):
    
    def test_str(self):
        var = Categorical("A")
        self.assertEqual(str(var), "A")
                
    def test_copy(self):
        orig = Categorical("A")
        copy = orig.copy()
        self.assertFalse(orig == copy)
        self.assertEqual(str(orig), str(copy))

    def test_interpret(self):
        var = Categorical("A")
        self.assertEqual(var.interpret(None), var)
                
class TestInteractionMethods(unittest.TestCase):
    
    def test_str(self):
        inter = Interaction(Quantitative("A"), Quantitative("B"))
        self.assertEqual(str(inter), "{A}{B}")
        
    def test_pow(self):
        inter = Interaction(Quantitative("A"), Quantitative("B"))
        inter = inter ** 3
        self.assertEqual(str(inter), "{A}{B}")
        with self.assertRaises(Exception):
            inter ** "log"
        
    def test_flatten(self):
        inter = Interaction(Quantitative("A"), Quantitative("B"))
        ls1 = inter.flatten()
        self.assertEqual(len(ls1), 1)
        self.assertEqual(ls1[0], inter)
        ls2 = inter.flatten(True)
        self.assertEqual(len(ls2), 2)
        self.assertEqual(str(ls2[0]), "A")
        self.assertEqual(str(ls2[1]), "B")
                
    def test_copy(self):
        orig = Interaction(Var("A"), Var("B"))
        copy = orig.copy()
        self.assertFalse(orig == copy)
        self.assertEqual(str(orig), str(copy))
        
    def test_interpret(self):
        old = Var("A") * Var("B")
        data = pd.DataFrame({"A" : [1], "B" : ["cat"]})
        old.interpret(data)
        self.assertTrue(isinstance(old.e1, Quantitative)
        self.assertTrue(isinstance(old.e2, Categorical)
                
class TestCombinationMethods(unittest.TestCase):
    
    def test_str(self):
        comb = Var("A") + Var("B")
        self.assertEqual(str(comb), "A + B")
        
    def test_mul(self):
        comb = Var("A") + Var("B")
        self.assertEqual(str(comb * Var("C")), "{A}{C} + {B}{C}")
        with self.assertRaises(Exception):
            comb * "bad arg"
        
    def test_imul(self):
        comb = Var("A") + Var("B")
        comb *= Var("C")
        self.assertEqual(str(comb), "{A}{C} + {B}{C}")
        with self.assertRaises(Exception):
            comb *= "bad arg"
            
    def test_pow(self):
        comb = Var("A") + Var("B")
        self.assertEqual(str(comb ** 2), "A^2 + {A}{B} + {B}{A} + B^2")
                
    def test_flatten(self):
        comb = Var("A") + Var("B")
        ls = comb.flatten()
        self.assertEqual(len(ls), 2)
        self.assertEqual(str(ls[0]), "A")
        self.assertEqual(str(ls[1]), "B")
                
    def test_copy(self):
        orig = Interaction(Var("A"), Var("B"))
        copy = orig.copy()
        self.assertFalse(orig == copy)
        self.assertEqual(str(orig), str(copy))
        
    def test_interpret(self):
        old = Var("A") + Var("B")
        data = pd.DataFrame({"A" : [1], "B" : ["cat"]})
        old.interpret(data)
        self.assertTrue(isinstance(old.e1, Quantitative)
        self.assertTrue(isinstance(old.e2, Categorical)
        
iris = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")


# Model        
class TestModelMethods(unittest.TestCase):
    
    def test_fit(self):
        levels = ["virginica", "setosa", "versicolor"]
        explanatory = Q("petal_width") + C("species", levels = levels)
        response = Q("sepal_width")
        model = LinearModel(explanatory, response)
        results = model.fit(iris)
        expected = pd.DataFrame({"Coefficients" : [1.36214962108944, 0.795582615454372, 1.86172822073969, 0.35290783081806], 
                                 "Standard Errors" : [0.248101683560026, 0.120657012161229, 0.223217037772943, 0.10358416791633], 
                                 "t-statistics" : [5.49, 6.59, 8.34, 3.41], 
                                 "p-values" : [0, 0, 0, 0.0008]}, 
                                 index = ["Intercept", "petal_width", "species::setosa", "species::verginica"])
        diff = results - expected
        diff["Coefficients"] = floatComparison(0, diff["Coefficients"], 0.000001)
        diff["Standard Errors"] = floatComparison(0, diff["Standard Errors"], 0.000001)
        diff["t-statistics"] = floatComparison(0, diff["t-statistics"], 0.01)
        diff["p-values"] = floatComparison(0, diff["p-values"], 0.0001)
        self.assertTrue(all(diff.apply(all, 1)))
        
    def test_predict(self):
        levels = ["virginica", "setosa", "versicolor"]
        explanatory = Q("petal_width") + C("species", levels = levels)
        response = Q("sepal_width")
        model = LinearModel(explanatory, response)
        model.fit(iris)
        newData = pd.DataFrame({"petal_width" : [1,2], "species" : ["setosa", "verginica"]})
        pred = model.predict(newData)
        expected = pd.DataFrame({"Predicted" : [4.019461, 3.306224]})
        diff = floatComparison(0, pred - expected, 0.00001)
        self.assertTrue(all(diff))
        
    def test_plot(self):
        model = LinearModel(Q("petal_wdith") + Q("petal_length"), Q("sepal_length")
        model.fit(iris)
        with self.assertRaises(Exception):
            model.plot()
                
    def test_residual_plots(self):
        model = LinearModel(Q("petal_wdith") + Q("petal_length"), Q("sepal_length")
        model.fit(iris)
        plots = model.residual_plots()
        self.assertEqual(len(plots), 2)
                
    def test_ones_column(self):
        ones = LinearModel.ones_column(iris)
        self.assertEqual(len(ones), 150)
        self.assertTrue(all(ones["Intercept"] == 1))
        
    '''        
    def test_extract_columns(self):
        self.assertEqual()
        with self.assertRaises(Exception):
            pass
        with self.assertRaises(Exception):
            pass
        with self.assertRaises(Exception):
            pass
        with self.assertRaises(Exception):
            pass
            
    def test_transform(self):
        self.assertEqual()
        with self.assertRaises(Exception):
            pass
        with self.assertRaises(Exception):
            pass            
    '''
    
if __name__ == "__main__":
    
        