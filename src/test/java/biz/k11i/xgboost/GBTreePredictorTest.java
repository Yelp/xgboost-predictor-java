package biz.k11i.xgboost;

import biz.k11i.xgboost.learner.ObjFunction;

import org.junit.After;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.FromDataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;

import java.io.IOException;

@RunWith(Theories.class)
public class GBTreePredictorTest extends PredictorTest {

    protected static final String MODEL_TYPE = "gbtree";

    @DataPoints("modelName")
    public static final String[] MODEL_NAMES = {
            "binary-logistic",
            "binary-logitraw",
            "multi-softmax",
            "multi-softprob",
    };

    @DataPoints("version")
    public static final String[] VERSIONS = { "40", "47" };

    @DataPoints
    public static final boolean[] USE_JAFAMA = { true, false };

    @Theory
    public void testPredict(
            @FromDataPoints("modelName") String modelName,
            @FromDataPoints("version") String version,
            boolean useJafama) throws IOException {

        ObjFunction.useFastMathExp(useJafama);
        String path = "model/" + MODEL_TYPE + "/" + modelNameWithVersion(version, modelName) + ".model";
        final Predictor predictor = newPredictor(path);

        verifyDouble(MODEL_TYPE, modelNameWithVersion(version, modelName), "predict",
            predictor::predict);

        verifyDouble(MODEL_TYPE, modelNameWithVersion(version, modelName), "predict_ntree",
            feat -> predictor.predict(feat, false, 1));

        verifyDouble(MODEL_TYPE, modelNameWithVersion(version, modelName), "margin",
            feat -> predictor.predict(feat, true));

        if (modelName.startsWith("binary-")) {
            // test predictSingle()
            verifyDouble(MODEL_TYPE, modelNameWithVersion(version, modelName), "predict",
                feat -> new double[] {predictor.predictSingle(feat)});
        }

        verifyInt(MODEL_TYPE, modelNameWithVersion(version, modelName), "leaf",
            predictor::predictLeaf);

        verifyInt(MODEL_TYPE, modelNameWithVersion(version, modelName), "leaf_ntree",
            feat -> predictor.predictLeaf(feat, 2));
    }

    @After
    public void tearDown() {
        ObjFunction.useFastMathExp(false);
    }
}