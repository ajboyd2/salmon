import unittest
from .expression import *
from .model import *
import pandas as pd

def floatComparison(a, b, eps = 0.0001):
    if isinstance(a, (pd.Series, pd.DataFrame)) or isinstance(b, (pd.Series, pd.DataFrame)):
        return (a - eps < b) & (b < a + eps)
    else:
        return (a - eps < b) and (b < a + eps)

# Expressions
class TestVarMethods(unittest.TestCase):
    
    def test_str(self):
        self.assertEqual(str(Var("Test Case")), "Test Case")
        
    def test_mul(self):
        self.assertEqual(str(Var("Var1") * Var("Var2")), "(Var1)(Var2)")
        self.assertEqual(str(Var("Test") * Poly("Single", 2)), "(Single)(Test)+(Single^2)(Test)")
        
    def test_imul(self):
        v = Var("Test")
        v *= Var("Other")
        self.assertEqual(str(v), "(Other)(Test)")
        v = Var("Test")
        v *= Poly("Single", 2)
        self.assertEqual(str(v), "(Single)(Test)+(Single^2)(Test)")
                
    def test_flatten(self):
        flattened = Var("Test").get_terms()
        self.assertEqual(len(flattened), 1)
        self.assertEqual(str(flattened[0]), "Test")
                
    def test_copy(self):
        orig = Var("A")
        copy = orig.copy()
        self.assertFalse(orig is copy)
        self.assertEqual(str(orig), str(copy))
                
    def test_interpret(self):
        data = pd.DataFrame({"A" : [1,2,3], "B" : ["hello", "test", "goodbye"]})
        var1 = Var("A")
        var2 = Var("B")
        self.assertTrue(isinstance(var1.interpret(data), Quantitative))
        self.assertTrue(isinstance(var2.interpret(data), Categorical))
                
class TestQuantitativeMethods(unittest.TestCase):
    
    def test_str(self):
        var = 4*(Quantitative("A")+3)**2
        self.assertEqual(str(var), "4*(6*A+9+A^2)")
        var = 4*Sin(Quantitative("A")+3)
        self.assertEqual(str(var), "4*sin(3+A)")
        
    def test_add(self):
        var1 = Quantitative("A")
        self.assertEqual(str(var1 + 3), "3+A")
        self.assertEqual(str(var1 + var1), "2*A")
        
    def test_iadd(self):
        var = Quantitative("A")
        var += 3
        self.assertEqual(str(var), "3+A")
        var += -3
        var += var
        self.assertEqual(str(var), "2*A")
                
    def test_mul(self):
        var = Quantitative("A")
        self.assertEqual(str(var * 3), "3*A")
        self.assertEqual(str(var * var), "A^2")
        self.assertEqual(str(var * Quantitative("B")), "(A)(B)")
                
    def test_imul(self):
        var1 = Quantitative("A")
        var1 *= 3
        self.assertEqual(str(var1), "3*A")
        var2 = Quantitative("A")
        var2 *= var2
        self.assertEqual(str(var2), "A^2")
        var3 = Quantitative("A")
        var3 *= Quantitative("B")
        self.assertEqual(str(var3), "(A)(B)")
        
    def test_pow(self):
        var = Quantitative("A")
        self.assertEqual(str(var ** 4), "A^4")
        with self.assertRaises(Exception):
            var ** "bad arg"
                
    def test_ipow(self):
        var = Quantitative("A")
        var **= 3
        self.assertEqual(str(var), "A^3")
        with self.assertRaises(Exception):
            var *= "bad arg"
                
    def test_transform(self):
        var = Quantitative("A")
        var = var.transform("sin")
        self.assertEqual(str(var), "sin(A)")
        with self.assertRaises(Exception):
            var = Quantitative("A")
            var = var ** 3
            var.transform("bad call")

    def test_copy(self):
        orig = Quantitative("A")
        copy = orig.copy()
        self.assertFalse(orig is copy)
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
        self.assertFalse(orig is copy)
        self.assertEqual(str(orig), str(copy))

    def test_interpret(self):
        var = Categorical("A")
        self.assertEqual(var.interpret(None), var)
                
class TestInteractionMethods(unittest.TestCase):
    
    def test_str(self):
        inter = Interaction(terms=(Quantitative("A"), Quantitative("B")))
        self.assertEqual(str(inter), "(A)(B)")
        
    def test_pow(self):
        inter = Interaction(terms=(Quantitative("A"), Quantitative("B")))
        inter = inter ** 3
        self.assertEqual(str(inter), "(A^3)(B^3)")
        with self.assertRaises(Exception):
            inter ** "log"
        
    def test_flatten(self):
        inter = Interaction(terms=(Quantitative("A"), Quantitative("B")))
        ls1 = inter.get_terms()
        self.assertEqual(len(ls1), 1)
        self.assertEqual(ls1[0], inter)
        ls2 = list(inter.terms)
        self.assertEqual(len(ls2), 2)
        self.assertTrue((str(ls2[0])=="A" and str(ls2[1])=="B") or (str(ls2[0])=="B" and str(ls2[1])=="A"))
                
    def test_copy(self):
        orig = Interaction(terms=(Var("A"), Var("B")))
        copy = orig.copy()
        self.assertFalse(orig is copy)
        self.assertEqual(str(orig), str(copy))
        
    def test_interpret(self):
        old = Var("A") * Var("B")
        data = pd.DataFrame({"A" : [1], "B" : ["cat"]})
        old.interpret(data)
        for v in old.terms:
            if v.name == "B":
                self.assertTrue(isinstance(v, Categorical))
            else:
                self.assertTrue(isinstance(v, Quantitative))
                
class TestCombinationMethods(unittest.TestCase):
    
    def test_str(self):
        comb = Var("A") + Var("B")
        self.assertEqual(str(comb), "A+B")
        
    def test_mul(self):
        comb = Var("A") + Var("B")
        self.assertEqual(str(comb * Var("C")), "(A)(C)+(B)(C)")
        with self.assertRaises(Exception):
            comb * "bad arg"
        
    def test_imul(self):
        comb = Var("A") + Var("B")
        comb *= Var("C")
        self.assertEqual(str(comb), "(A)(C)+(B)(C)")
        with self.assertRaises(Exception):
            comb *= "bad arg"
            
    def test_pow(self):
        comb = Var("A") + Var("B")
        self.assertEqual(str(comb ** 2), "2*(A)(B)+A^2+B^2")
                
    def test_flatten(self):
        comb = Var("A") + Var("B")
        ls = list(comb.get_terms())
        self.assertEqual(len(ls), 2)
        self.assertTrue((str(ls[0])=="A" and str(ls[1])=="B") or (str(ls[0])=="B" and str(ls[1])=="A"))

                
    def test_copy(self):
        orig = Interaction(terms=(Var("A"), Var("B")))
        copy = orig.copy()
        self.assertFalse(orig is copy)
        self.assertEqual(str(orig), str(copy))
        
    def test_interpret(self):
        old = Var("A") + Var("B")
        data = pd.DataFrame({"A" : [1], "B" : ["cat"]})
        old.interpret(data)
        for v in old.get_terms():
            if v.name == "B":
                self.assertTrue(isinstance(v, Categorical))
            else:
                self.assertTrue(isinstance(v, Quantitative))
        
iris = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")


# Model        
class TestModelMethods(unittest.TestCase):
    
    def test_fit(self):
        levels = ["virginica", "setosa", "versicolor"]
        explanatory = Q("petal_width") + C("species", levels = levels)
        response = Q("sepal_width")
        model = LinearModel(explanatory, response)
        results = model.fit(iris)[["Coefficient", "SE", "t", "p"]].sort_index()
        expected = pd.DataFrame({"Coefficient" : [1.36214962108944, 0.795582615454372, 1.86172822073969, 0.35290783081806], 
                                 "SE" : [0.248101683560026, 0.120657012161229, 0.223217037772943, 0.10358416791633], 
                                 "t" : [5.49, 6.59, 8.34, 3.41], 
                                 "p" : [0, 0, 0, 0.0008]}, 
                                 index = ["Intercept", "petal_width", "species{setosa}", "species{versicolor}"]).sort_index()
        diff = results - expected
        diff["Coefficient"] = floatComparison(0, diff["Coefficient"], 0.000001)
        diff["SE"] = floatComparison(0, diff["SE"], 0.000001)
        diff["t"] = floatComparison(0, diff["t"], 0.01)
        diff["p"] = floatComparison(0, diff["p"], 0.0001)

        self.assertTrue(all(diff.apply(all, 1)))
        
    def test_predict(self):
        levels = ["virginica", "setosa", "versicolor"]
        explanatory = Q("petal_width") + C("species", levels=levels)
        response = Q("sepal_width")
        model = LinearModel(explanatory, response)
        model.fit(iris)
        newData = pd.DataFrame({"petal_width" : [1,2], "species" : ["setosa", "virginica"]})
        pred = model.predict(newData)
        expected = pd.DataFrame({"Predicted" : [4.019461, 3.306224]})
        diff = floatComparison(0, pred - expected, 0.00001)
        self.assertTrue(all(diff))
        
    def test_plot(self):
        model = LinearModel(Q("petal_width") + Q("petal_length"), Q("sepal_length"))
        model.fit(iris)
        with self.assertRaises(Exception):
            model.plot()
                
    def test_residual_plots(self):
        model = LinearModel(Q("petal_width") + Q("petal_length"), Q("sepal_length"))
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
    unittest.main()
        
