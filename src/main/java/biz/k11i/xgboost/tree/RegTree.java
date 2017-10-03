package biz.k11i.xgboost.tree;

import biz.k11i.xgboost.util.FVec;

/**
 * Regression tree.
 *
 * Memory block model.
 *
 * A node is composed of 3 blocks of ints:
 *
 *   ------------------------------------------------------
 * 1 |       Split condition / Leaf Value (32 bits)       |
 *   ------------------------------------------------------
 * 2 | Left Child Address (32 bits; 0  iff node is leaf)  |
 *   ------------------------------------------------------
 * 3 | Feature Index (31 bits) | Default is right (1-bit) |
 *   ------------------------------------------------------
 *
 * Design Limitations:
 * - Since we allocate a full int for the left child address, the restriction on tree size is JVM
 * specific but always < Integer.MAX_VALUE. Therefore, a complete tree can never exceed depth 31.
 * Hypothetically we will never reach this limit. Because of this, the left child address field
 * is pre-multiplied by BLOCK_SIZE to reduce future computation.
 *
 * - Feature index is allocated 31 bits. This implementation can only support 2^31 features.
 * Unlike with node indexes, we use longs for our sparse feature maps and therefore could
 * possibly support ~2.1 billion features.
 */
public class RegTree extends AbstractRegTree {
  public int[] nodes;
  public static final int BLOCK_SIZE = 3;

  @Override
  public void loadModel(Param param) {
    nodes = new int[BLOCK_SIZE * param.num_nodes];
    for (int i = 0; i < BLOCK_SIZE * param.num_nodes; i += BLOCK_SIZE) {
      Node node = param.nodeInfo[i / BLOCK_SIZE];
      /*
       * Store node attributes in contiguous memory. Use Bit masks to store and read attributes.
       */
      nodes[i] = createNodeValue(node);
      nodes[i + 1] = createNodeChildren(node);
      nodes[i + 2] = createNodeDefaultAndValue(node);
    }
  }

  public int createNodeValue(Node nodeObj) {
    if (nodeObj._isLeaf) {
      return Float.floatToRawIntBits(nodeObj.leaf_value);
    }
    return Float.floatToRawIntBits(nodeObj.split_cond);
  }

  public int createNodeChildren(Node nodeObj) {
    if (nodeObj.is_leaf()) {
      return 0;
    } else {
      return nodeObj.cleft_ * BLOCK_SIZE;
    }
  }

  public int createNodeDefaultAndValue(Node nodeObj) {
    return (nodeObj.split_index() << 1) | (nodeObj.default_left() ? 0 : 1);
  }

  @Override
  public int getNextNode(int index, FVec feat) {
    Number fvalue = feat.fvalue(getFeatureIndex(nodes[index + 2]));

    if (null == fvalue) {
      if (isDefaultLeft(nodes[index + 2])) {
        return nodes[index + 1];
      } else {
        return nodes[index + 1] + BLOCK_SIZE;
      }
    }

    if (fvalue.doubleValue() < Float.intBitsToFloat(nodes[index])) {
      return nodes[index + 1];
    } else {
      return nodes[index + 1] + BLOCK_SIZE;
    }
  }

  @Override
  public double getLeafValue(int node) {
    return Float.intBitsToFloat(nodes[node]);
  }

  @Override
  protected int getLeafIndex(int node) {
    return node / BLOCK_SIZE;
  }

  @Override
  public boolean isLeafNode(int node) {
    return nodes[node + 1] == 0;
  }

  public static int getLeftChild(int node) {
    return node;
  }

  public static int getRightChild(int node) {
    return node + BLOCK_SIZE;
  }

  public static int getFeatureIndex(int node) {
    return node >>> 1;
  }

  public static boolean isDefaultLeft(int node) {
    return (node & 1) == 0;
  }
}
