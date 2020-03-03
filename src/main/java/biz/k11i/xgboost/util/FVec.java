package biz.k11i.xgboost.util;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Interface of feature vector.
 *
 * This class was modified from the open source version it was forked from
 * to always store feature values as floats to resolve
 * https://github.com/komiya-atsushi/xgboost-predictor-java/issues/21
 */
public interface FVec extends Serializable {
    /**
     * Gets index-th value.
     *
     * @param index index
     * @return value
     */
    Float fvalue(int index);

    class Transformer {
        private Transformer() {
            // do nothing
        }

        /**
         * Builds FVec from dense vector.
         *
         * @param values         float values
         * @param treatsZeroAsNA treat zero as N/A if true
         * @return FVec
         */
        public static FVec fromArray(float[] values, boolean treatsZeroAsNA) {
            return new FVecArrayImpl.FVecFloatArrayImpl(values, treatsZeroAsNA);
        }

        /**
         * Builds FVec from dense vector.
         *
         * @param values         double values
         * @param treatsZeroAsNA treat zero as N/A if true
         * @return FVec
         */
        public static FVec fromArray(double[] values, boolean treatsZeroAsNA) {
            float[] float_values = new float[values.length];
            for (int i = 0; i < values.length; i++)
            {
                float_values[i] = (float) values[i];
            }
            return new FVecArrayImpl.FVecFloatArrayImpl(float_values, treatsZeroAsNA);
        }

        /**
         * Builds FVec from map.
         *
         * @param map map containing non-zero values
         * @return FVec
         */
        public static FVec fromMap(Map<Integer, ? extends Number> map) {
            Map<Integer, Float> floatMap = new HashMap<>();

            for (Map.Entry<Integer, ? extends Number> indexAndValue: map.entrySet()) {
                floatMap.put(indexAndValue.getKey(), indexAndValue.getValue().floatValue());
            }
            return new FVecMapImpl(floatMap);
        }
    }

    class FVecMapImpl implements FVec {
        private final Map<Integer, ? extends Float> values;

        FVecMapImpl(Map<Integer, Float> values) {
            this.values = values;
        }

        @Override
        public Float fvalue(int index) {
            return values.get(index);
        }
    }

    class FVecArrayImpl {
        static class FVecFloatArrayImpl implements FVec {
            private final float[] values;
            private final boolean treatsZeroAsNA;

            FVecFloatArrayImpl(float[] values, boolean treatsZeroAsNA) {
                this.values = values;
                this.treatsZeroAsNA = treatsZeroAsNA;
            }

            @Override
            public Float fvalue(int index) {
                if (values.length <= index) {
                    return Float.NaN;
                }

                Float result = values[index];
                if (treatsZeroAsNA && result == 0) {
                    return Float.NaN;
                }

                return result;
            }
        }
    }
}
