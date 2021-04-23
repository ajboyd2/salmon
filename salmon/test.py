import unittest
# from .expression import *
# from .model import *
import pandas as pd


def floatComparison(a, b, eps=0.0001):
    if isinstance(a, (pd.Series, pd.DataFrame)) or isinstance(
            b, (pd.Series, pd.DataFrame)):
        return (a - eps < b) & (b < a + eps)
    else:
        return (a - eps < b) and (b < a + eps)

# Expressions


class TestVarMethods(unittest.TestCase):

    def test_str(self):
        self.assertEqual(str(Var("Test Case")), "Test Case")

    def test_mul(self):
        self.assertEqual(str(Var("Var1") * Var("Var2")), "(Var1)(Var2)")
        self.assertEqual(str(Var("Test") * Poly("Single", 2)),
                         "(Single)(Test)+(Single^2)(Test)")

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
        data = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["hello", "test", "goodbye"]})
        var1 = Var("A")
        var2 = Var("B")
        self.assertTrue(isinstance(var1.interpret(data), Quantitative))
        self.assertTrue(isinstance(var2.interpret(data), Categorical))


class TestQuantitativeMethods(unittest.TestCase):

    def test_str(self):
        var = 4 * (Quantitative("A") + 3)**2
        self.assertEqual(str(var), "4*(6*A+9+A^2)")
        var = 4 * Sin(Quantitative("A") + 3)
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
        self.assertTrue((str(ls2[0]) == "A" and str(ls2[1]) == "B") or (
            str(ls2[0]) == "B" and str(ls2[1]) == "A"))

    def test_copy(self):
        orig = Interaction(terms=(Var("A"), Var("B")))
        copy = orig.copy()
        self.assertFalse(orig is copy)
        self.assertEqual(str(orig), str(copy))

    def test_interpret(self):
        old = Var("A") * Var("B")
        data = pd.DataFrame({"A": [1], "B": ["cat"]})
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
        self.assertTrue((str(ls[0]) == "A" and str(ls[1]) == "B") or (
            str(ls[0]) == "B" and str(ls[1]) == "A"))

    def test_copy(self):
        orig = Interaction(terms=(Var("A"), Var("B")))
        copy = orig.copy()
        self.assertFalse(orig is copy)
        self.assertEqual(str(orig), str(copy))

    def test_interpret(self):
        old = Var("A") + Var("B")
        data = pd.DataFrame({"A": [1], "B": ["cat"]})
        old.interpret(data)
        for v in old.get_terms():
            if v.name == "B":
                self.assertTrue(isinstance(v, Categorical))
            else:
                self.assertTrue(isinstance(v, Quantitative))


iris = pd.read_csv("data/iris.csv")
commprop = pd.read_csv("data/CommProp.csv")
realestate = pd.read_csv("data/Real Estate5.csv")
plastic = pd.read_csv("data/Plastic.csv")

# Model


class TestModelMethods(unittest.TestCase):

    def test_fit(self):
        levels = ["virginica", "setosa", "versicolor"]
        explanatory = Q("petal_width") + C("species", levels=levels)
        response = Q("sepal_width")
        model = LinearModel(explanatory, response)
        results = model.fit(iris)[["Coefficient", "SE", "t", "p"]].sort_index()
        expected = pd.DataFrame({"Coefficient": [1.36214962108944,
                                                 0.795582615454372,
                                                 1.86172822073969,
                                                 0.35290783081806],
                                 "SE": [0.248101683560026, 0.120657012161229,
                                        0.223217037772943, 0.10358416791633],
                                 "t": [5.49, 6.59, 8.34, 3.41],
                                 "p": [0, 0, 0, 0.0008]},
                                index=["Intercept", "petal_width",
                                       "species{setosa}",
                                       "species{versicolor}"]).sort_index()
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
        newData = pd.DataFrame(
            {"petal_width": [1, 2], "species": ["setosa", "verginica"]})
        pred = model.predict(newData)
        expected = pd.DataFrame({"Predicted": [4.019461, 3.306224]})
        diff = floatComparison(0, pred - expected, 0.00001)
        self.assertTrue(all(diff))

    def test_plot(self):
        model = LinearModel(
            Q("petal_width") +
            Q("petal_length"),
            Q("sepal_length"))
        model.fit(iris)
        with self.assertRaises(Exception):
            model.plot()

    def test_residual_plots(self):
        model = LinearModel(
            Q("petal_width") +
            Q("petal_length"),
            Q("sepal_length"))
        model.fit(iris)
        plots = model.residual_plots()
        self.assertEqual(len(plots), 2)

    def test_ones_column(self):
        ones = LinearModel.ones_column(iris)
        self.assertEqual(len(ones), 150)
        self.assertTrue(all(ones["Intercept"] == 1))

    def test_fit2(self):
        explanatory = Q("Age") + Q("Expenses") + Q("Vacancy") + Q("Sqft")
        response = Q("Rental")
        model = LinearModel(explanatory, response)
        results = model.fit(commprop)[
            ["Coefficient", "SE", "t", "p"]].sort_index()
        expected = pd.DataFrame({"Coefficient": [12.20, -1.420336e-01,
                                                 2.820165e-01, 6.193435e-03,
                                                 7.924302e-06],
                                 "SE": [5.779562e-01, 2.134261e-02,
                                        6.317235e-02, 1.086813e-02,
                                        1.384775e-06],
                                 "t": [21.1098807, -6.6549332, 4.4642400,
                                       0.5698714, 5.7224457],
                                 "p": [1.601720e-33, 3.894322e-09,
                                       2.747396e-05, 5.704457e-01,
                                       1.975990e-07]},
                                index=["Intercept", "Age", "Expenses",
                                       "Vacancy", "Sqft"]).sort_index()
        diff = results - expected
        diff["Coefficient"] = floatComparison(0, diff["Coefficient"], 0.001)
        diff["SE"] = floatComparison(0, diff["SE"], 0.000001)
        diff["t"] = floatComparison(0, diff["t"], 0.01)
        diff["p"] = floatComparison(0, diff["p"], 0.0001)

        self.assertTrue(all(diff.apply(all, 1)))

    def test_predict_prediction2(self):
        explanatory = Q("Age") + Q("Expenses") + Q("Vacancy") + Q("Sqft")
        response = Q("Rental")
        model = LinearModel(explanatory, response)
        model.fit(commprop)
        newData = pd.DataFrame({"Age": [8, 9], "Expenses": [7.50, 5.75],
                                "Vacancy": [10, 18], "Sqft": [140000, 127500]})
        pred = model.predict(newData, prediction_interval=.05)
        expected = pd.DataFrame({"Predicted": [14.35078, 13.66571],
                                 "2.5%": [12.05951, 11.3438],
                                 "97.5%": [16.64205, 15.98762]})
        diff = floatComparison(0, pred - expected, 0.00001)

        self.assertTrue(all(diff))

    def test_predict_confidence2(self):
        explanatory = Q("Age") + Q("Expenses") + Q("Vacancy") + Q("Sqft")
        response = Q("Rental")
        model = LinearModel(explanatory, response)
        model.fit(commprop)
        newData = pd.DataFrame({"Age": [8, 9], "Expenses": [7.50, 5.75],
                                "Vacancy": [10, 18], "Sqft": [140000, 127500]})
        pred = model.predict(newData, confidence_interval=.05)
        expected = pd.DataFrame({"Predicted": [14.35078, 13.66571],
                                 "2.5%": [14.00029, 14.70126],
                                 "97.5%": [13.15169, 14.17972]})
        diff = floatComparison(0, pred - expected, 0.00001)

        self.assertTrue(all(diff))

    def test_confidence2(self):
        explanatory = Q("Age") + Q("Expenses") + Q("Vacancy") + Q("Sqft")
        response = Q("Rental")
        model = LinearModel(explanatory, response)
        model.fit(commprop)
        confidence = model.confidence_intervals()
        expected = pd.DataFrame({"2.5%": [1.104949e+01, -1.845411e-01,
                                          1.561979e-01, -1.545232e-02,
                                          5.166283e-06],
                                 "97.5%": [1.335169e+01, -9.952615e-02,
                                           4.078352e-01, 2.783919e-02,
                                           1.068232e-05]},
                                index=["Intercept", "Age", "Expenses",
                                       "Vacancy", "Sqft"]).sort_index()
        diff = floatComparison(0, confidence - expected, 0.00001)

        self.assertTrue(all(diff))

    def test_funcs2(self):
        explanatory = Q("Age") + Q("Expenses") + Q("Vacancy") + Q("Sqft")
        response = Q("Rental")
        model = LinearModel(explanatory, response)
        model.fit(commprop)
        self.assertAlmostEqual(model.get_ssr(), 138.3269, 4)
        self.assertAlmostEqual(model.get_sse(), 98.23059, 4)
        self.assertAlmostEqual(model.get_sst(), 236.5575, 4)
        self.assertAlmostEqual(model.r_squared(), 0.5847496, 6)
        self.assertAlmostEqual(model.r_squared(adjusted=True), 0.5628943, 6)

    def test_fit3(self):
        exp = Q("Time")
        resp = Q("Hardness")
        model = LinearModel(exp, resp)
        results = model.fit(plastic)[
            ["Coefficient", "SE", "t", "p"]].sort_index()
        expected = pd.DataFrame({"Coefficient": [168.600000, 2.034375],
                                 "SE": [2.65702405, 0.09039379],
                                 "t": [63.45445, 22.50569],
                                 "p": [1.257946e-18, 2.158814e-12]},
                                index=["Intercept", "Time"]).sort_index()
        diff = results - expected
        diff["Coefficient"] = floatComparison(0, diff["Coefficient"], 0.000001)
        diff["SE"] = floatComparison(0, diff["SE"], 0.000001)
        diff["t"] = floatComparison(0, diff["t"], 0.01)
        diff["p"] = floatComparison(0, diff["p"], 0.0001)

        self.assertTrue(all(diff.apply(all, 1)))

    def test_predict_prediction3(self):
        exp = Q("Time")
        resp = Q("Hardness")
        model = LinearModel(exp, resp)
        model.fit(plastic)
        newData = pd.DataFrame({"Time": [18, 20, 30, 33, 37]})
        pred = model.predict(newData, prediction_interval=.02)
        expected = pd.DataFrame({"Predicted": [205.2187, 209.2875,
                                               229.6312, 235.7344,
                                               243.8719],
                                 "1%": [196.1539, 200.3351, 220.8695,
                                        226.9054, 234.8662],
                                 "99%": [214.2836, 218.2399, 238.393,
                                         244.5633, 252.8775]})
        diff = floatComparison(0, pred - expected, 0.00001)

        self.assertTrue(all(diff))

    def test_predict_confidence3(self):
        exp = Q("Time")
        resp = Q("Hardness")
        model = LinearModel(exp, resp)
        model.fit(plastic)
        newData = pd.DataFrame({"Time": [18, 20, 30, 33, 37]})
        pred = model.predict(newData, confidence_interval=.02)
        expected = pd.DataFrame({"Predicted": [205.2187, 209.2875,
                                               229.6312, 235.7344,
                                               243.8719],
                                 "1%": [202.0359, 206.4406, 227.4569,
                                        233.3034, 240.8617],
                                 "99%": [208.4016, 212.1344, 231.8056,
                                         238.1653, 246.8821]})
        diff = floatComparison(0, pred - expected, 0.00001)

        self.assertTrue(all(diff))

    def test_confidence3(self):
        exp = Q("Time")
        resp = Q("Hardness")
        model = LinearModel(exp, resp)
        model.fit(plastic)
        confidence = model.confidence_intervals(alpha=0.02)
        expected = pd.DataFrame({"1%": [161.626656, 1.797137],
                                 "99%": [175.573344, 2.271613]},
                                index=["Intercept", "Time"]).sort_index()
        diff = floatComparison(0, confidence - expected, 0.00001)

        self.assertTrue(all(diff))

    def test_funcs3(self):
        exp = Q("Time")
        resp = Q("Hardness")
        model = LinearModel(exp, resp)
        model.fit(plastic)
        self.assertAlmostEqual(model.get_ssr(), 5297.512, 2)
        self.assertAlmostEqual(model.get_sse(), 146.425, 2)
        self.assertAlmostEqual(model.get_sst(), 5443.937, 2)
        self.assertAlmostEqual(model.r_squared(), 0.9731031, 6)
        self.assertAlmostEqual(model.r_squared(adjusted=True), 0.9711819, 6)

    def test_fit4(self):
        exp = Q("Log2Sqft") + Q("Bed") + Q("Bath") + Q("Age") + \
            C("Quality", levels=["Medium", "High", "Low"])
        resp = Q("Log2Price")
        model = LinearModel(exp, resp)
        results = model.fit(realestate)[
            ["Coefficient", "SE", "t", "p"]].sort_index()
        expected = pd.DataFrame({"Coefficient": [9.623520955, 0.747909265,
                                                 0.005029163, 0.060444837,
                                                 -0.004838176, 0.480246914,
                                                 -0.125872691],
                                 "SE": [0.5118051486, 0.0492714657,
                                        0.0145788831, 0.0190479692,
                                        0.0008426408, 0.0434530465,
                                        0.0349050618],
                                 "t": [18.8030952, 15.1793590,
                                       0.3449622, 3.1732956,
                                       -5.7416826, 11.0520885,
                                       -3.6061443],
                                 "p": [1.948280e-60, 2.799404e-43,
                                       7.302636e-01, 1.597210e-03,
                                       1.603338e-08, 1.257162e-25,
                                       3.410388e-04]},
                                index=["Intercept", "Log2Sqft",
                                       "Bed", "Bath", "Age",
                                       "Quality{High}",
                                       "Quality{Low}"]).sort_index()
        diff = results - expected
        diff["Coefficient"] = floatComparison(0, diff["Coefficient"], 0.000001)
        diff["SE"] = floatComparison(0, diff["SE"], 0.000001)
        diff["t"] = floatComparison(0, diff["t"], 0.01)
        diff["p"] = floatComparison(0, diff["p"], 0.0001)

        self.assertTrue(all(diff.apply(all, 1)))

    def test_predict_prediction4(self):
        exp = Q("Log2Sqft") + Q("Bed") + Q("Bath") + Q("Age") + \
            C("Quality", levels=["Medium", "High", "Low"])
        resp = Q("Log2Price")
        model = LinearModel(exp, resp)
        model.fit(realestate)
        newData = pd.DataFrame({"Log2Sqft": [12, 11, 10],
                                "Bed": [3, 2, 4], "Bath": [2, 3, 5],
                                "Age": [25, 12, 9],
                                "Quality": ["Medium", "High", "Low"]})
        pred = model.predict(newData, prediction_interval=.01)
        expected = pd.DataFrame({"Predicted": [18.61345, 18.4641, 17.25554],
                                 "0.5%": [17.91869, 17.7756, 16.53876],
                                 "99.5%": [19.30822, 19.15261, 17.97231]})
        diff = floatComparison(0, pred - expected, 0.00001)

        self.assertTrue(all(diff))

    def test_predict_confidence4(self):
        exp = Q("Log2Sqft") + Q("Bed") + Q("Bath") + Q("Age") + \
            C("Quality", levels=["Medium", "High", "Low"])
        resp = Q("Log2Price")
        model = LinearModel(exp, resp)
        model.fit(realestate)
        newData = pd.DataFrame({"Log2Sqft": [12, 11, 10],
                                "Bed": [3, 2, 4], "Bath": [2, 3, 5],
                                "Age": [25, 12, 9],
                                "Quality": ["Medium", "High", "Low"]})
        pred = model.predict(newData, confidence_interval=.01)
        expected = pd.DataFrame({"Predicted": [18.61345, 18.4641, 17.25554],
                                 "0.5%": [18.46566, 18.34925, 17.02551],
                                 "99.5%": [18.76125, 18.57896, 17.48556]})
        diff = floatComparison(0, pred - expected, 0.00001)

        self.assertTrue(all(diff))

    def test_confidence4(self):
        exp = Q("Log2Sqft") + Q("Bed") + Q("Bath") + Q("Age") + \
            C("Quality", levels=["Medium", "High", "Low"])
        resp = Q("Log2Price")
        model = LinearModel(exp, resp)
        model.fit(realestate)
        confidence = model.confidence_intervals()
        expected = pd.DataFrame({"2.5%": [8.618038286, 0.651111480,
                                          -0.023612233, 0.023023559,
                                          -0.006493612, 0.394879885,
                                          -0.194446512],
                                 "97.5%": [10.62900362, 0.84470705,
                                           0.03367056, 0.09786612,
                                           -0.00318274, 0.56561394,
                                           -0.05729887]},
                                index=["Intercept", "Log2Sqft", "Bed",
                                       "Bath", "Age", "Quality{High}",
                                       "Quality{Low}"]).sort_index()
        diff = floatComparison(0, confidence - expected, 0.00001)

        self.assertTrue(all(diff))

    def test_funcs4(self):
        exp = Q("Log2Sqft") + Q("Bed") + Q("Bath") + Q("Age") + \
            C("Quality", levels=["Medium", "High", "Low"])
        resp = Q("Log2Price")
        model = LinearModel(exp, resp)
        model.fit(realestate)
        self.assertAlmostEqual(model.get_ssr(), 166.5586, 3)
        self.assertAlmostEqual(model.get_sse(), 35.50675, 4)
        self.assertAlmostEqual(model.get_sst(), 202.0654, 3)
        self.assertAlmostEqual(model.r_squared(), 0.8242809, 6)
        self.assertAlmostEqual(model.r_squared(adjusted=True), 0.8222337, 6)

    def test_fit5(self):
        exp = Poly("Age", 2) + C("Quality", levels=["Medium", "High", "Low"])
        resp = Q("Log2Price")
        model = LinearModel(exp, resp)
        results = model.fit(realestate)[
            ["Coefficient", "SE", "t", "p"]].sort_index()
        expected = pd.DataFrame({"Coefficient": [18.4292586578, -0.0196314047,
                                                 0.0001579584, 0.8384196551,
                                                 -0.4893179786],
                                 "SE": [6.682494e-02, 3.154817e-03,
                                        3.370501e-05, 5.451302e-02,
                                        4.049609e-02],
                                 "t": [275.784124, -6.222675,
                                       4.686496, 15.380171,
                                       -12.083092],
                                 "p": [0.000000e+00, 1.010611e-09,
                                       3.558999e-06, 3.152482e-44,
                                       8.841368e-30]},
                                index=["Intercept", "Age", "Age^2",
                                       "Quality{High}",
                                       "Quality{Low}"]).sort_index()
        diff = results - expected
        diff["Coefficient"] = floatComparison(0, diff["Coefficient"], 0.000001)
        diff["SE"] = floatComparison(0, diff["SE"], 0.000001)
        diff["t"] = floatComparison(0, diff["t"], 0.01)
        diff["p"] = floatComparison(0, diff["p"], 0.0001)

        self.assertTrue(all(diff.apply(all, 1)))

    def test_funcs5(self):
        exp = Poly("Age", 2) + C("Quality", levels=["Medium", "High", "Low"])
        resp = Q("Log2Price")
        model = LinearModel(exp, resp)
        model.fit(realestate)
        self.assertAlmostEqual(model.get_ssr(), 136.1445, 3)
        self.assertAlmostEqual(model.get_sse(), 65.9209, 4)
        self.assertAlmostEqual(model.get_sst(), 202.0654, 3)
        self.assertAlmostEqual(model.r_squared(), 0.6737645, 6)
        self.assertAlmostEqual(model.r_squared(adjusted=True), 0.6712404, 6)

    def test_confidence5(self):
        exp = Poly("Age", 2) + C("Quality", levels=["Medium", "High", "Low"])
        resp = Q("Log2Price")
        model = LinearModel(exp, resp)
        model.fit(realestate)
        confidence = model.confidence_intervals(alpha=0.1)
        expected = pd.DataFrame({"5%": [18.3191440957, -0.0248299327,
                                        0.0001024191, 0.7485927516,
                                        -0.5560476897],
                                 "95%": [18.5393732198, -0.0144328766,
                                         0.0002134977, 0.9282465586,
                                         -0.4225882675]},
                                index=["Intercept", "Age", "Age^2",
                                       "Quality{High}",
                                       "Quality{Low}"]).sort_index()
        diff = floatComparison(0, confidence - expected, 0.00001)

        self.assertTrue(all(diff))

    def test_fit6(self):
        level = ["Medium", "High", "Low"]
        exp = Q("Age") + C("Quality", levels=level) + \
            Q("Age") * C("Quality", levels=level)
        resp = Q("Log2Price")
        model = LinearModel(exp, resp)
        results = model.fit(realestate)[
            ["Coefficient", "SE", "t", "p"]].sort_index()
        expected = pd.DataFrame({"Coefficient": [18.2242342193, -0.0071330947,
                                                 0.8977071766, -0.6794336205,
                                                 0.0004980208, 0.0039311280],
                                 "SE": [0.051956912, 0.001502665,
                                        0.091080634, 0.116414599,
                                        0.004068962, 0.002526017],
                                 "t": [350.756683, -4.746961,
                                       9.856181, -5.836327,
                                       0.122395, 1.556255],
                                 "p": [0.000000e+00, 2.678341e-06,
                                       4.107532e-21, 9.437550e-09,
                                       9.026338e-01, 1.202605e-01]},
                                index=["Intercept", "Age",
                                       "Quality{High}", "Quality{Low}",
                                       "(Age)(Quality{High})",
                                       "(Age)(Quality{Low})"]).sort_index()
        diff = results - expected
        diff["Coefficient"] = floatComparison(0, diff["Coefficient"],
                                              0.000001)
        diff["SE"] = floatComparison(0, diff["SE"], 0.000001)
        diff["t"] = floatComparison(0, diff["t"], 0.01)
        diff["p"] = floatComparison(0, diff["p"], 0.0001)

        self.assertTrue(all(diff.apply(all, 1)))

    def test_confidence6(self):
        exp = Poly("Age", 2) + C("Quality", levels=["Medium", "High", "Low"])
        resp = Q("Log2Price")
        model = LinearModel(exp, resp)
        model.fit(realestate)
        confidence = model.confidence_intervals(alpha=0.1)
        expected = pd.DataFrame({"2.5%": [18.122161123, -0.010085189,
                                          0.718772710, -0.908138486,
                                          -0.007495749, -0.001031415],
                                 "97.5%": [18.326307316, -0.004181000,
                                           1.076641643, -0.450728755,
                                           0.008491790, 0.008893671]},
                                index=["Intercept", "Age",
                                       "Quality{High}", "Quality{Low}",
                                       "(Age)(Quality{High})",
                                       "(Age)(Quality{Low})"]).sort_index()
        diff = floatComparison(0, confidence - expected, 0.00001)

        self.assertTrue(all(diff))

    def test_funcs6(self):
        level = ["Medium", "High", "Low"]
        exp = Q("Age") + C("Quality", levels=level) + \
            Q("Age") * C("Quality", levels=level)
        resp = Q("Log2Price")
        model = LinearModel(exp, resp)
        model.fit(realestate)
        self.assertAlmostEqual(model.get_ssr(), 133.6717, 3)
        self.assertAlmostEqual(model.get_sse(), 68.39363, 4)
        self.assertAlmostEqual(model.get_sst(), 202.0653, 3)
        self.assertAlmostEqual(model.r_squared(), 0.6615272, 6)
        self.assertAlmostEqual(model.r_squared(adjusted=True), 0.6582474, 6)

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
