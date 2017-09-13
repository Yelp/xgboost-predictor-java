package biz.k11i.xgboost.tree;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;

import biz.k11i.xgboost.util.FVec;

/**
 * Repacked regression tree.
 *
 * Memory block model.
 *
 * A node is composed of 3 blocks of ints:
 *
 *   ------------------------------------------------------
 * 1 |       Split condition / Leaf Value (32 bits)       |
 *   ------------------------------------------------------
 * 2 |  Right Child Offset (31 bits) | No is left (1 bit) | (Zero iff node is leaf)
 *   ------------------------------------------------------
 * 3 | Feature Index (31 bits) | Default is right (1 bit) |
 *   ------------------------------------------------------
 *
 * Design Specifics / Limitations:
 * - XGBoost trees are indexed breadth-first from the root with the left child always immediately
 * adjacent to the right child. We repack the tree with respect to node coverage giving the
 * property that frequently evaluated paths are compact in memory, in theory encouraging cache
 * hits. It is always the case that a parent node has a higher cover than a child node, but not
 * that a node has a higher cover than a child to a sibling node. The trade-off is that rare
 * routes almost guarantee a cache miss.
 *
 * - Array indices are 32 bit integers whose max value is JVM specific but always < Integer
 * .MAX_VALUE. Additionally, since tree nodes are packed with stride BLOCK_SIZE, this upper bound
 * is effectively further reduced by log2(3). We overcome this limitation by using an otherwise
 * wasted sign bit of the int for determining conditional direction and further by considering
 * the child offset rather than outright index, which in a complete tree has an upper bound of
 * exp(2, node_depth). Therefore we require 2 fewer bits than a simple int indexing.
 *
 * - Feature index is allocated 31 bits. This implementation can only support 2^31 features.
 * Unlike with node indexes, we use longs for our sparse feature vectors and therefore could
 * only possibly support ~2.1 billion features.
 */
public class RepackedRegTree extends AbstractRegTree {
  private static final int BLOCK_SIZE = 3;

  private int[] nodes;

  @Override
  public void loadModel(Param param) {
    int i = 0;
    nodes = new int[BLOCK_SIZE * param.num_nodes];

    ArrayDeque<Node> stack = new ArrayDeque<>();
    Map<Integer, Integer> newIndexMap = new HashMap<>(param.num_nodes);

    stack.add(param.nodeInfo[0]);

    // Performs a depth-first iteration breaking ties by cover to add Nodes to the node int array
    while (!stack.isEmpty()) {
      Node current = stack.removeLast();

      newIndexMap.put(current.id, i);

      nodes[i] = createNodeValue(current);
      nodes[i + 2] = createNodeDefaultAndValue(current);

      if (current.is_leaf()) {
        nodes[i + 1] = 0x0;
      } else {
        Node left = param.nodeInfo[current.cleft_];
        Node right = param.nodeInfo[current.cright_];

        // Note: stores a 1 in right child offset as a placeholder to distinguish from a the 0x0
        // stored by a leaf in the case that the original left is still left.
        if (left.sum_hess > right.sum_hess) {
          stack.addLast(right);
          stack.addLast(left);
          nodes[i + 1] = 0b10;
        } else {
          stack.addLast(left);
          stack.addLast(right);
          nodes[i + 1] = 0b11;
          nodes[i + 2] ^= 0x1; // Flips the default path since left/right have been flipped
        }
      }

      i += BLOCK_SIZE;
    }

    // Once all nodes have been added to the int array, update offsets to right children
    for (Node node : param.nodeInfo) {
      if (!node.is_leaf()) {
        int parentId = newIndexMap.get(node.id);

        int distantChildId = (nodes[parentId + 1] & 0x1) == 0 ? node.cright_: node.cleft_;
        int newChildId = newIndexMap.get(distantChildId);
        nodes[parentId + 1] = ((newChildId - parentId) << 1) ^ (nodes[parentId + 1] & 0x1);
      }
    }
  }

  @Override
  protected int getNextNode(int index, FVec feat) {
    Number fvalue = feat.fvalue(nodes[index + 2] >>> 1);

    if (null == fvalue) {
      if ((nodes[index + 2] & 1) == 0) {
        return index + BLOCK_SIZE;
      } else {
        return index + (nodes[index + 1] >>> 1);
      }
    }

    if (
        (fvalue.doubleValue() < Float.intBitsToFloat(nodes[index])) !=
            ((nodes[index + 1] & 0x1) == 1)
        ) {
      return index + BLOCK_SIZE;
    } else {
      return index + (nodes[index + 1] >>> 1);
    }
  }

  @Override
  protected boolean isLeafNode(int node) {
    return nodes[node + 1] == 0;
  }

  @Override
  protected double getLeafValue(int node) {
    return Float.intBitsToFloat(nodes[node]);
  }

  private int createNodeValue(Node node) {
    if (node._isLeaf) {
      return Float.floatToRawIntBits(node.leaf_value);
    }
    return Float.floatToRawIntBits(node.split_cond);
  }

  public int createNodeDefaultAndValue(Node node) {
    return (node.split_index() << 1) | (node.default_left() ? 0 : 1);
  }
}
