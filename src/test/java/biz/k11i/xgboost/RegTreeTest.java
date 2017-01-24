package biz.k11i.xgboost;
import biz.k11i.xgboost.tree.RegTree;

import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class RegTreeTest {
  RegTree regTree = new RegTree();

  @Test(expected = IOException.class)
  public void treeExceedsMaxLength() throws IOException {
    RegTree.Node node = mock(RegTree.Node.class);
    node.cright_ = 0x1ffff;
    regTree.createNodeChildren(node);
  }

  @Test
  public void createNodeChildrenTest() throws IOException {
    RegTree.Node node = mock(RegTree.Node.class);
    node.cright_ = 0x11;
    node.cleft_ = 0x12;
    int children = regTree.createNodeChildren(node);
    assertEquals((children >>> 16) & RegTree.QUAD_WORD, node.cleft_);
    assertEquals(children & RegTree.QUAD_WORD, node.cright_);
  }

  @Test
  public void createNodeLeafDefaultAndValueLeafTest() {
    RegTree.Node node = mock(RegTree.Node.class);
    node._isLeaf = true;
    when(node.split_index()).thenReturn(0x5);
    int nodeValue = regTree.createNodeLeafDefaultAndValue(node);
    assertEquals(nodeValue & RegTree.LEAF_MASK, RegTree.LEAF_MASK);
    assertEquals(nodeValue & RegTree.SPLIT_MASK, 0x5);
  }

  @Test
  public void createNodeLeafDefaultAndValueNonLeafTest() {
    RegTree.Node node = mock(RegTree.Node.class);
    node._isLeaf = false;
    when(node.default_left()).thenReturn(true);
    int nodeValue = regTree.createNodeLeafDefaultAndValue(node);
    assertEquals(nodeValue & RegTree.DEFAULT_MASK, RegTree.DEFAULT_MASK);
  }

  @Test
  public void getLeftChildTest() {
    int node = 0xffff0000;
    assertEquals(RegTree.getLeftChild(node), 0xffff * 3);
    node = 0xffffaaaa;
    assertEquals(RegTree.getLeftChild(node), 0xffff * 3);
    node = 0xafffaaaa;
    assertEquals(RegTree.getLeftChild(node), 0xafff * 3);
  }


  @Test
  public void getRightChildTest() {
    int node = 0xffff0000;
    assertEquals(RegTree.getRightChild(node), 0);
    node = 0xffffaaaa;
    assertEquals(RegTree.getRightChild(node), 0xaaaa * 3);
    node = 0x0000aaa1;
    assertEquals(RegTree.getRightChild(node), 0xaaa1 * 3);
  }


  @Test
  public void getFeatureIndexTest() {
    int node = 0x3fff0000;
    assertEquals(RegTree.getFeatureIndex(node), 0x3fff0000);
    node = 0xffffaaaa;
    assertEquals(RegTree.getFeatureIndex(node), 0x3fffaaaa);
    node = 0x0000000a;
    assertEquals(RegTree.getFeatureIndex(node), 0xa);
  }


  @Test
  public void getDefaultAndLeafTest() {
    int node = 0x3fff0000;
    assertEquals(RegTree.isDefaultLeft(node), false);
    assertEquals(RegTree.isNotLeaf(node), true);
    node = 0xffffaaaa;
    assertEquals(RegTree.isDefaultLeft(node), true);
    assertEquals(RegTree.isNotLeaf(node), false);
    node = 0xafffaaaa;
    assertEquals(RegTree.isDefaultLeft(node), false);
    assertEquals(RegTree.isNotLeaf(node), false);
  }
}
