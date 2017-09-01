package biz.k11i.xgboost;
import biz.k11i.xgboost.tree.RegTree;

import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class RegTreeTest {
  RegTree regTree = new RegTree();

  @Test
  public void createNodeChildrenTest() throws IOException {
    RegTree.Node node = mock(RegTree.Node.class);
    node.cleft_ = 0x11;
    node.cright_ = 0x12;
    int children = regTree.createNodeChildren(node);
    assertEquals(children, node.cleft_ * RegTree.BLOCK_SIZE);
  }

  @Test
  public void createNodeDefaultAndValueLeafTest() {
    RegTree.Node node = mock(RegTree.Node.class);
    when(node.split_index()).thenReturn(0x5);
    int nodeValue = regTree.createNodeDefaultAndValue(node);
    assertEquals(nodeValue >>> 1, 0x5);
  }

  @Test
  public void createNodeDefaultAndValueNonLeafTest() {
    RegTree.Node node = mock(RegTree.Node.class);
    node._isLeaf = false;
    when(node.default_left()).thenReturn(true);
    int nodeValue = regTree.createNodeDefaultAndValue(node);
    assertEquals(nodeValue & 1, 0x0);
  }


  @Test
  public void getRightChildTest() {
    int node = 0xffff0000;
    assertEquals(RegTree.getRightChild(node), node + 3);
    node = 0xffffaaaa;
    assertEquals(RegTree.getRightChild(node), node + 3);
    node = 0x0000aaa1;
    assertEquals(RegTree.getRightChild(node), node + 3);
  }


  @Test
  public void getFeatureIndexTest() {
    int node = 0x3fff0000;
    assertEquals(RegTree.getFeatureIndex(node), node >>> 1);
  }


  @Test
  public void getDefaultAndLeafTest() {
    int node = 0x3fff0000;
    assertEquals(RegTree.isDefaultLeft(node), true);
    assertEquals(RegTree.isNotLeaf(node), true);
    node = 0xffffaaab;
    assertEquals(RegTree.isDefaultLeft(node), false);
    assertEquals(RegTree.isNotLeaf(node), true);
    node = 0xafffaaaa;
    assertEquals(RegTree.isDefaultLeft(node), true);
    assertEquals(RegTree.isNotLeaf(node), true);
    node = 0x0;
    assertEquals(RegTree.isDefaultLeft(node), true);
    assertEquals(RegTree.isNotLeaf(node), false);
  }
}
