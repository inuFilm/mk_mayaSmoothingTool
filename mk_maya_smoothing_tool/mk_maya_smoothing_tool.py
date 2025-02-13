"""
Maya Smoothing Tool (Static Update Version with Undo, Redo, Freeze Border, 0X Freeze, and Border Vertex Unselect)

【概要】
- シーン内にオブジェクトが存在しない場合は警告を表示します。
- ウィンドウ起動時に対象メッシュの全頂点位置（base state）、隣接情報、および
  ユーザー提供のロジックにより取得した境界頂点（Freeze Border 対象）をキャプチャします。
- ラジオボタンで3種の手法（Taubin Smoothing, Laplacian Smoothing, Volume Preserving Smoothing）を選択可能。
- 各手法ごとに各種パラメータ（Smooth Amount, Iterations, Tension, など）をスライダー＋SpinBoxで設定可能です。
- Execute ボタン押下時、base state から平滑化処理を実行し、結果を反映（Undo/Redo 対応、頂点選択状態復元）。
- Freeze Border チェックがONの場合、ユーザー提供のロジックで取得した境界頂点は平滑化対象外になります。
- 0X Freeze チェックがONの場合、ワールドX座標がほぼ0の頂点は更新されません。
- 選択が頂点以外の場合も、自動的に頂点コンポーネントに変換して処理を行います。
- さらに「Border Vertex Unselect」ボタンにより、現在の選択から境界頂点を除外（非境界頂点のみ選択）できます。
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

# ── 境界頂点の取得（ユーザー提供のロジック） ─────────────────────────────
def getBorderVerticesFromMesh(meshName):
    # オブジェクトのシェイプノードを取得
    shapes = cmds.listRelatives(meshName, shapes=True, fullPath=True) or []
    if not shapes:
        return set()
    shape = shapes[0]
    # 全フェースを取得
    faces = cmds.ls(shape + ".f[*]", flatten=True)
    if not faces:
        return set()
    # 境界エッジのみを抽出
    border_edges = cmds.polyListComponentConversion(faces, toEdge=True, border=True)
    if not border_edges:
        return set()
    border_edges = cmds.ls(border_edges, flatten=True)
    if not border_edges:
        return set()
    # 境界エッジから頂点に変換
    border_verts = cmds.polyListComponentConversion(border_edges, toVertex=True)
    if not border_verts:
        return set()
    border_verts = cmds.ls(border_verts, flatten=True)
    indices = set()
    for v in border_verts:
        try:
            idx = int(v[v.find('[')+1 : v.find(']')])
            indices.add(idx)
        except:
            pass
    return indices

# ── メッシュデータ取得 ─────────────────────────────
def getSelectedVertices():
    sels = cmds.ls(selection=True)
    if not sels:
        cmds.warning("何も選択されていません。")
        return None, []
    expanded = cmds.filterExpand(sels, sm=31)
    if not expanded:
        converted = cmds.polyListComponentConversion(sels, toVertex=True)
        expanded = cmds.filterExpand(converted, sm=31)
    if not expanded:
        cmds.warning("頂点コンポーネントが見つかりません。")
        return None, []
    meshName = expanded[0].split('.')[0]
    indices = []
    for v in expanded:
        try:
            i = int(v[v.find('[')+1 : v.find(']')])
            indices.append(i)
        except:
            pass
    return meshName, list(set(indices))

def getMeshData(meshName):
    if not cmds.ls(geometry=True):
        cmds.warning("シーン内にオブジェクトが存在しません。\nオブジェクトを配置してください。")
        return None, None, None
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
    for e in edges:
        info = cmds.polyInfo(e, edgeToVertex=True)
        if info:
            parts = info[0].strip().split()
            try:
                v1 = int(parts[-2])
                v2 = int(parts[-1])
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)
            except:
                pass
    borderVertices = getBorderVerticesFromMesh(meshName)
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
def taubinSmooth(positions, adjacency, borderVertices, selIndices, lambda_val, mu_val, tension, freeze_border, sharp_edge_border, freezeX, iterations):
    newPos = positions[:]  # shallow copy
    for it in range(iterations):
        temp = newPos[:]
        for i in selIndices:
            if freeze_border and (i in borderVertices):
                continue
            if freezeX and abs(temp[i][0]) < 1e-6:
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
            if freezeX and abs(temp[i][0]) < 1e-6:
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

def laplacianSmoothBasic(positions, adjacency, borderVertices, selIndices, smooth_amount, alpha, tension, freeze_border, sharp_edge_border, freezeX, iterations):
    newPos = positions[:]
    original = {i: positions[i][:] for i in selIndices}
    for it in range(iterations):
        temp = newPos[:]
        for i in selIndices:
            if freeze_border and (i in borderVertices):
                continue
            if freezeX and abs(temp[i][0]) < 1e-6:
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

# ── 現在の選択から境界頂点を除外する機能 ─────────────────────────────
def unselectBorderVertices():
    meshName, selIndices = getSelectedVertices()
    if not meshName:
        cmds.warning("何も選択されていません。")
        return
    borderIndices = getBorderVerticesFromMesh(meshName)
    newIndices = list(set(selIndices) - borderIndices)
    if not newIndices:
        cmds.warning("境界頂点以外の頂点が選択されていません。")
        cmds.select(clear=True)
        return
    shapes = cmds.listRelatives(meshName, shapes=True, fullPath=True) or [meshName]
    shape = shapes[0]
    newSel = [f"{shape}.vtx[{i}]" for i in newIndices]
    cmds.select(newSel, replace=True)
    print("Border Vertex Unselect：境界頂点を選択解除しました。")

# ── UI用スライダーウィジェット ─────────────────────────────
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
        
        if not cmds.ls(geometry=True):
            QtWidgets.QMessageBox.warning(self, "Warning", "シーン内にオブジェクトが存在しません。\nオブジェクトを配置してください。")
            self.meshName = None
            return
        
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
        
        # 共通の0X Freezeチェックボックス（スタックの下）
        self.freezeX = QtWidgets.QCheckBox("0X Freeze (ワールドX=0の頂点は固定)")
        mainLayout.addWidget(self.freezeX)
        
        # Execute ボタン
        self.btnExecute = QtWidgets.QPushButton("Execute")
        self.btnExecute.clicked.connect(self.onExecute)
        mainLayout.addWidget(self.btnExecute)
        
        # Border Vertex Unselect ボタン（現在の選択から境界頂点を除外）
        self.btnBorderUnselect = QtWidgets.QPushButton("Border Vertex Unselect")
        self.btnBorderUnselect.clicked.connect(unselectBorderVertices)
        mainLayout.addWidget(self.btnBorderUnselect)
        
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

        layout.addWidget(self.taubin_smooth_amount)
        layout.addWidget(self.taubin_iteration)
        layout.addLayout(tensionLayout)
        layout.addWidget(self.taubin_lambda)
        layout.addWidget(self.taubin_mu)
        layout.addWidget(self.taubin_sharp)

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
            freezeX = self.freezeX.isChecked()
            if method == 0:  # Taubin Smoothing
                lam = self.taubin_lambda.value() * self.taubin_smooth_amount.value()
                mu  = -self.taubin_mu.value() * self.taubin_smooth_amount.value()
                newPositions = taubinSmooth(positions, adjacency, borderVertices, selIndices, lam, mu, 
                                              self.taubin_tension.currentText(),
                                              self.taubin_freeze.isChecked(), self.taubin_sharp.value(),
                                              freezeX, self.taubin_iteration.value())
            elif method == 1:  # Laplacian Smoothing
                newPositions = laplacianSmoothBasic(positions, adjacency, borderVertices, selIndices, 
                                                    self.lap_smooth_amount.value(), self.lap_alpha.value(),
                                                    self.lap_tension.currentText(),
                                                    self.lap_freeze.isChecked(), self.lap_sharp.value(),
                                                    freezeX, self.lap_iteration.value())
            elif method == 2:  # Volume Preserving Smoothing
                lam = self.vol_smooth_amount.value() * 0.35
                mu  = -self.vol_smooth_amount.value() * 0.36
                newPositions = taubinSmooth(positions, adjacency, borderVertices, selIndices, lam, mu, 
                                              self.vol_tension.currentText(),
                                              self.vol_freeze.isChecked(), self.vol_sharp.value(),
                                              freezeX, self.vol_iteration.value())
            setMeshPositions(meshName, newPositions)
            compList = [f"{meshName}.vtx[{i}]" for i in selIndices]
            cmds.select(compList, replace=True)
            print("Execute：平滑化処理が完了しました。")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
        finally:
            cmds.undoInfo(closeChunk=True)  # Undo チャンク終了
            
def unselectBorderVertices():
    meshName, selIndices = getSelectedVertices()
    if not meshName:
        cmds.warning("何も選択されていません。")
        return
    borderIndices = getBorderVerticesFromMesh(meshName)
    newIndices = list(set(selIndices) - borderIndices)
    if not newIndices:
        cmds.warning("境界頂点以外の頂点が選択されていません。")
        cmds.select(clear=True)
        return
    shapes = cmds.listRelatives(meshName, shapes=True, fullPath=True) or [meshName]
    shape = shapes[0]
    newSel = [f"{shape}.vtx[{i}]" for i in newIndices]
    cmds.select(newSel, replace=True)
    print("Border Vertex Unselect：境界頂点を選択解除しました。")

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
