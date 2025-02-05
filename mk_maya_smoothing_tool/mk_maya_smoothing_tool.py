"""
openAI chatGPT o3mini-high使用
MK Maya Smoothing Tool (Static Update Version with Undo, Redo, and Freeze Border Correction)

【概要】
- ウィンドウ起動時に対象メッシュの全頂点位置（base state）、隣接情報、及び境界頂点（borderVertices）をキャプチャ
- ラジオボタンで以下の3種の手法を選択可能：
    [Taubin Smoothing] / [Laplacian Smoothing] / [Volume Preserving Smoothing]
- 各手法ごとに、各種パラメータをスライダー＋SpinBoxで設定可能
- Execute ボタン押下時に、base state から平滑化処理を実行し、結果を反映（頂点選択状態は復元）
- Undo/Redo ボタンで、直前の平滑化操作を Maya の Undo/Redo で戻せます

※ Taubin smoothing は λ‐ステップ＋μ‐ステップによる体積保持を狙った手法です。
※ Laplacian Smoothing は単純なラプラシアン平滑化＋元位置ブレンドです。
※ Volume Preserving Smoothing は Taubin smoothing に近い処理として実装しています。
※ Freeze Border は、getMeshData() 内で算出した境界頂点に対して平滑化処理を適用しないチェックです。
"""

import maya.cmds as cmds
import math
from collections import defaultdict

# ── ベクトル演算ヘルパー ─────────────────────────────
def addVec(a, b):
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def subVec(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def mulVec(a, s):
    return [a[0]*s, a[1]*s, a[2]*s]

def divVec(a, s):
    if abs(s) < 1e-8:
        return a
    return [a[0]/s, a[1]/s, a[2]/s]

# ── 隣接情報・重み計算 ─────────────────────────────
def computeWeight(p, q, tension):
    d = math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2)
    if tension == "UNIFORM":
        return 1.0
    elif tension == "INVERSE":
        return 1.0 / max(d, 1e-8)
    elif tension == "PROPORTIONAL":
        return d
    return 1.0

# ── メッシュデータ取得 ─────────────────────────────
def getSelectedVertices():
    sel = cmds.filterExpand(sm=31)  # 頂点コンポーネント
    if not sel:
        sel = cmds.ls(cmds.ls(selection=True)[0] + ".vtx[*]", flatten=True)
    meshName = sel[0].split('.')[0]
    indices = []
    for v in sel:
        try:
            i = int(v[v.find('[')+1 : v.find(']')])
            indices.append(i)
        except:
            pass
    return meshName, list(set(indices))

def getMeshData(meshName):
    """
    メッシュ名から、全頂点位置、隣接情報、及び境界頂点セットを返す。
    境界頂点は、各エッジの edgeToFace 情報を用い、面数が 1 のエッジに接続する頂点を抽出。
    """
    shapes = cmds.listRelatives(meshName, shapes=True, fullPath=True) or []
    if not shapes:
        cmds.error("メッシュシェイプが見つかりません。")
    shape = shapes[0]
    vertexCount = cmds.polyEvaluate(shape, vertex=True)
    positions = []
    for i in range(vertexCount):
        pos = cmds.pointPosition(f"{shape}.vtx[{i}]", world=True)
        positions.append(pos)
    
    edges = cmds.polyListComponentConversion(f"{shape}.vtx[*]", toEdge=True)
    edges = cmds.filterExpand(edges, sm=32) or []
    adjacency = defaultdict(set)
    borderVertices = set()
    for e in edges:
        info_v = cmds.polyInfo(e, edgeToVertex=True)
        if info_v:
            parts_v = info_v[0].strip().split()
            try:
                v1 = int(parts_v[-2])
                v2 = int(parts_v[-1])
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)
            except:
                pass
        info_f = cmds.polyInfo(e, edgeToFace=True)
        if info_f:
            parts_f = info_f[0].strip().split()
            faceCount = len(parts_f) - 2
            if faceCount < 2:
                if info_v:
                    try:
                        borderVertices.add(int(parts_v[-2]))
                        borderVertices.add(int(parts_v[-1]))
                    except:
                        pass
    return positions, adjacency, borderVertices

def setMeshPositions(meshName, positions):
    shapes = cmds.listRelatives(meshName, shapes=True, fullPath=True) or []
    if not shapes:
        cmds.error("メッシュシェイプが見つかりません。")
    shape = shapes[0]
    vertexCount = cmds.polyEvaluate(shape, vertex=True)
    if vertexCount != len(positions):
        cmds.error("頂点数が一致しません。")
    for i, pos in enumerate(positions):
        cmds.xform(f"{shape}.vtx[{i}]", worldSpace=True, translation=pos)
    cmds.refresh(f=True)

# ── 平滑化アルゴリズム ─────────────────────────────
def taubinSmooth(positions, adjacency, borderVertices, selIndices, lambda_val, mu_val, tension, freeze_border, sharp_edge_border, iterations):
    newPos = positions[:]  # shallow copy
    for it in range(iterations):
        temp = newPos[:]
        for i in selIndices:
            if freeze_border and (i in borderVertices):
                continue
            delta = [0.0, 0.0, 0.0]
            weightSum = 0.0
            for nb in adjacency[i]:
                w = computeWeight(temp[i], temp[nb], tension)
                delta = addVec(delta, mulVec(subVec(temp[nb], temp[i]), w))
                weightSum += w
            if weightSum > 0:
                delta = divVec(delta, weightSum)
            if len(adjacency[i]) < 3:
                factor = 1.0 - (sharp_edge_border / 100.0)
                delta = mulVec(delta, factor)
            newPos[i] = addVec(temp[i], mulVec(delta, lambda_val))
        temp = newPos[:]
        for i in selIndices:
            if freeze_border and (i in borderVertices):
                continue
            delta = [0.0, 0.0, 0.0]
            weightSum = 0.0
            for nb in adjacency[i]:
                w = computeWeight(temp[i], temp[nb], tension)
                delta = addVec(delta, mulVec(subVec(temp[nb], temp[i]), w))
                weightSum += w
            if weightSum > 0:
                delta = divVec(delta, weightSum)
            if len(adjacency[i]) < 3:
                factor = 1.0 - (sharp_edge_border / 100.0)
                delta = mulVec(delta, factor)
            newPos[i] = addVec(temp[i], mulVec(delta, mu_val))
    return newPos

def laplacianSmoothBasic(positions, adjacency, borderVertices, selIndices, smooth_amount, alpha, tension, freeze_border, sharp_edge_border, iterations):
    newPos = positions[:]
    original = {i: positions[i][:] for i in selIndices}
    for it in range(iterations):
        temp = newPos[:]
        for i in selIndices:
            if freeze_border and (i in borderVertices):
                continue
            delta = [0.0, 0.0, 0.0]
            weightSum = 0.0
            for nb in adjacency[i]:
                w = computeWeight(temp[i], temp[nb], tension)
                delta = addVec(delta, mulVec(subVec(temp[nb], temp[i]), w))
                weightSum += w
            if weightSum > 0:
                delta = divVec(delta, weightSum)
            new_pt = addVec(temp[i], mulVec(delta, smooth_amount))
            blended = [ (1 - alpha) * new_pt[j] + alpha * original[i][j] for j in range(3) ]
            if len(adjacency[i]) < 3:
                factor = 1.0 - (sharp_edge_border / 100.0)
                blended = [ temp[i][j] + (blended[j]-temp[i][j]) * factor for j in range(3) ]
            newPos[i] = blended
    return newPos

# ── UI用スライダーウィジェット（スライダー＋SpinBox） ─────────────────────────────
from PySide2 import QtWidgets, QtCore
import maya.OpenMayaUI as omui
from shiboken2 import wrapInstance

class SliderWidget(QtWidgets.QWidget):
    def __init__(self, label, min_val, max_val, initial, is_float=True, decimals=2, parent=None):
        super(SliderWidget, self).__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        self.label = QtWidgets.QLabel(label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.is_float = is_float
        if is_float:
            self.slider.setMinimum(0)
            self.slider.setMaximum(1000)
            self.slider.setValue(int(initial * 1000))
        else:
            self.slider.setMinimum(min_val)
            self.slider.setMaximum(max_val)
            self.slider.setValue(initial)
        self.spin = QtWidgets.QDoubleSpinBox() if is_float else QtWidgets.QSpinBox()
        self.spin.setMinimum(min_val)
        self.spin.setMaximum(max_val)
        self.spin.setValue(initial)
        if is_float:
            self.spin.setDecimals(decimals)
            self.spin.setSingleStep((max_val - min_val) / 100.0)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)
        self.slider.valueChanged.connect(self.onSliderChanged)
        self.spin.valueChanged.connect(self.onSpinChanged)
    def onSliderChanged(self, value):
        if self.is_float:
            newVal = value / 1000.0
        else:
            newVal = value
        self.spin.blockSignals(True)
        self.spin.setValue(newVal)
        self.spin.blockSignals(False)
    def onSpinChanged(self, value):
        if self.is_float:
            self.slider.blockSignals(True)
            self.slider.setValue(int(value * 1000))
            self.slider.blockSignals(False)
    def value(self):
        return self.spin.value()

# ── Mayaウィンドウ取得 ─────────────────────────────
def getMayaWindow():
    ptr = omui.MQtUtil.mainWindow()
    if ptr is not None:
        return wrapInstance(int(ptr), QtWidgets.QWidget)
    return None

# ── UIウィンドウ（静的更新版） ─────────────────────────────
class SmoothingToolWindow(QtWidgets.QDialog):
    def __init__(self, parent=getMayaWindow()):
        super(SmoothingToolWindow, self).__init__(parent)
        self.setWindowTitle("Smoothing Tool")
        self.setMinimumWidth(450)
        self.buildUI()
        
        # ウィンドウ起動時に対象メッシュのベース状態、隣接情報、境界頂点をキャプチャ
        meshName, selIndices = getSelectedVertices()
        if meshName:
            positions, adjacency, borderVertices = getMeshData(meshName)
            self.meshName = meshName
            self.selIndices = selIndices if selIndices else list(range(len(positions)))
            self.basePositions = positions[:]  # base state
            self.adjacency = adjacency
            self.borderVertices = borderVertices
        else:
            self.meshName = None
        
    def buildUI(self):
        mainLayout = QtWidgets.QVBoxLayout(self)
        
        # ラジオボタン群（横並び）
        self.methodGroup = QtWidgets.QButtonGroup(self)
        radioLayout = QtWidgets.QHBoxLayout()
        self.radioTaubin = QtWidgets.QRadioButton("Taubin Smoothing")
        self.radioLaplacian = QtWidgets.QRadioButton("Laplacian Smoothing")
        self.radioVolumePreserving = QtWidgets.QRadioButton("Volume Preserving Smoothing")
        self.methodGroup.addButton(self.radioTaubin, 0)
        self.methodGroup.addButton(self.radioLaplacian, 1)
        self.methodGroup.addButton(self.radioVolumePreserving, 2)
        self.radioVolumePreserving.setChecked(True)
        radioLayout.addWidget(self.radioTaubin)
        radioLayout.addWidget(self.radioLaplacian)
        radioLayout.addWidget(self.radioVolumePreserving)
        mainLayout.addLayout(radioLayout)
        
        # スタックウィジェット（各手法のパラメータ）
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.buildTaubinPage())
        self.stack.addWidget(self.buildLaplacianPage())
        self.stack.addWidget(self.buildVolumePage())
        mainLayout.addWidget(self.stack)
        
        self.methodGroup.buttonClicked[int].connect(self.stack.setCurrentIndex)
        
        # Execute ボタン
        self.btnExecute = QtWidgets.QPushButton("Execute")
        self.btnExecute.clicked.connect(self.onExecute)
        mainLayout.addWidget(self.btnExecute)
        
        # Undo/Redo ボタン群
        btnLayout = QtWidgets.QHBoxLayout()
        self.btnUndo = QtWidgets.QPushButton("Undo")
        self.btnUndo.clicked.connect(lambda: cmds.undo())
        self.btnRedo = QtWidgets.QPushButton("Redo")
        self.btnRedo.clicked.connect(lambda: cmds.redo())
        btnLayout.addWidget(self.btnUndo)
        btnLayout.addWidget(self.btnRedo)
        mainLayout.addLayout(btnLayout)
        
    def buildTaubinPage(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.taubin_smooth_amount = SliderWidget("Smooth Amount:", 0.0, 1.0, 1.0, is_float=True, decimals=2)
        self.taubin_iteration = SliderWidget("Iterations:", 1, 20, 5, is_float=False)
        tensionLayout = QtWidgets.QHBoxLayout()
        tensionLabel = QtWidgets.QLabel("Tension:")
        self.taubin_tension = QtWidgets.QComboBox()
        self.taubin_tension.addItems(["INVERSE", "PROPORTIONAL", "UNIFORM"])
        self.taubin_tension.setCurrentText("UNIFORM")
        tensionLayout.addWidget(tensionLabel)
        tensionLayout.addWidget(self.taubin_tension)
        self.taubin_lambda = SliderWidget("Lambda Factor:", 0.0, 1.0, 0.33, is_float=True, decimals=2)
        self.taubin_mu = SliderWidget("Mu Factor:", 0.0, 1.0, 0.34, is_float=True, decimals=2)
        self.taubin_sharp = SliderWidget("Sharp Edge Border:", 0, 100, 1, is_float=False)
        self.taubin_freeze = QtWidgets.QCheckBox("Freeze Border")
        layout.addWidget(self.taubin_smooth_amount)
        layout.addWidget(self.taubin_iteration)
        layout.addLayout(tensionLayout)
        layout.addWidget(self.taubin_lambda)
        layout.addWidget(self.taubin_mu)
        layout.addWidget(self.taubin_sharp)
        layout.addWidget(self.taubin_freeze)
        return page

    def buildLaplacianPage(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.lap_smooth_amount = SliderWidget("Smooth Amount:", 0.0, 1.0, 1.0, is_float=True, decimals=2)
        self.lap_iteration = SliderWidget("Iterations:", 1, 20, 5, is_float=False)
        tensionLayout = QtWidgets.QHBoxLayout()
        tensionLabel = QtWidgets.QLabel("Tension:")
        self.lap_tension = QtWidgets.QComboBox()
        self.lap_tension.addItems(["INVERSE", "PROPORTIONAL", "UNIFORM"])
        self.lap_tension.setCurrentText("UNIFORM")
        tensionLayout.addWidget(tensionLabel)
        tensionLayout.addWidget(self.lap_tension)
        self.lap_alpha = SliderWidget("Alpha:", 0.0, 1.0, 0.5, is_float=True, decimals=2)
        self.lap_sharp = SliderWidget("Sharp Edge Border:", 0, 100, 1, is_float=False)
        self.lap_freeze = QtWidgets.QCheckBox("Freeze Border")
        layout.addWidget(self.lap_smooth_amount)
        layout.addWidget(self.lap_iteration)
        layout.addLayout(tensionLayout)
        layout.addWidget(self.lap_alpha)
        layout.addWidget(self.lap_sharp)
        layout.addWidget(self.lap_freeze)
        return page

    def buildVolumePage(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.vol_smooth_amount = SliderWidget("Smooth Amount:", 0.0, 1.0, 1.0, is_float=True, decimals=2)
        self.vol_iteration = SliderWidget("Iterations:", 1, 20, 5, is_float=False)
        tensionLayout = QtWidgets.QHBoxLayout()
        tensionLabel = QtWidgets.QLabel("Tension:")
        self.vol_tension = QtWidgets.QComboBox()
        self.vol_tension.addItems(["INVERSE", "PROPORTIONAL", "UNIFORM"])
        self.vol_tension.setCurrentText("UNIFORM")
        tensionLayout.addWidget(tensionLabel)
        tensionLayout.addWidget(self.vol_tension)
        self.vol_normal_smooth = SliderWidget("Normal Smooth:", 0.0, 1.0, 0.5, is_float=True, decimals=2)
        self.vol_inflate = SliderWidget("Inflate:", -2.0, 2.0, 0.85, is_float=True, decimals=2)
        self.vol_sharp = SliderWidget("Sharp Edge Border:", 0, 100, 1, is_float=False)
        self.vol_freeze = QtWidgets.QCheckBox("Freeze Border")
        layout.addWidget(self.vol_smooth_amount)
        layout.addWidget(self.vol_iteration)
        layout.addLayout(tensionLayout)
        layout.addWidget(self.vol_normal_smooth)
        layout.addWidget(self.vol_inflate)
        layout.addWidget(self.vol_sharp)
        layout.addWidget(self.vol_freeze)
        return page

    def onExecute(self):
        try:
            cmds.undoInfo(openChunk=True)  # Undo チャンク開始
            meshName, selIndices = getSelectedVertices()
            if not meshName:
                return
            positions, adjacency, borderVertices = getMeshData(meshName)
            if not selIndices:
                selIndices = list(range(len(positions)))
            method = self.methodGroup.checkedId()
            if method == 0:  # Taubin Smoothing
                lam = self.taubin_lambda.value() * self.taubin_smooth_amount.value()
                mu  = -self.taubin_mu.value() * self.taubin_smooth_amount.value()
                newPositions = taubinSmooth(positions, adjacency, borderVertices, selIndices, lam, mu, 
                                              self.taubin_tension.currentText(),
                                              self.taubin_freeze.isChecked(), self.taubin_sharp.value(),
                                              self.taubin_iteration.value())
            elif method == 1:  # Laplacian Smoothing
                newPositions = laplacianSmoothBasic(positions, adjacency, borderVertices, selIndices, 
                                                    self.lap_smooth_amount.value(), self.lap_alpha.value(),
                                                    self.lap_tension.currentText(),
                                                    self.lap_freeze.isChecked(), self.lap_sharp.value(),
                                                    self.lap_iteration.value())
            elif method == 2:  # Volume Preserving Smoothing
                lam = self.vol_smooth_amount.value() * 0.35
                mu  = -self.vol_smooth_amount.value() * 0.36
                newPositions = taubinSmooth(positions, adjacency, borderVertices, selIndices, lam, mu, 
                                              self.vol_tension.currentText(),
                                              self.vol_freeze.isChecked(), self.vol_sharp.value(),
                                              self.vol_iteration.value())
            setMeshPositions(meshName, newPositions)
            compList = [f"{meshName}.vtx[{i}]" for i in selIndices]
            cmds.select(compList, replace=True)
            print("Execute：平滑化処理が完了しました。")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
        finally:
            cmds.undoInfo(closeChunk=True)  # Undo チャンク終了
            
def showSmoothingTool():
    global smoothingToolWindow
    try:
        smoothingToolWindow.close()
        smoothingToolWindow.deleteLater()
    except:
        pass
    smoothingToolWindow = SmoothingToolWindow()
    smoothingToolWindow.show()

# ツール表示
showSmoothingTool()
