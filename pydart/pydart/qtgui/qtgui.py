import os
import sys
import signal
import time
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from glwidget import GLWidget
from trackball import Trackball
from OpenGL.GLUT import glutInit


def signal_handler(signal, frame):
    print 'You pressed Ctrl+C! Bye.'
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


class PyDartQtWindow(QtWidgets.QMainWindow):
    def __init__(self, _title, _sim, _render):
        super(PyDartQtWindow, self).__init__()

        # Check and create captures directory
        if not os.path.isdir('captures'):
            os.makedirs('captures')

        self.sim = _sim
        self.setWindowTitle(_title)
        self.trackball = None
        self.render = _render

        self.initUI()
        self.initActions()
        self.initToolbarActions()
        self.initToolbar()
        self.initMenu()

        self.idleTimer = QtCore.QTimer()
        self.idleTimer.timeout.connect(self.idleTimerEvent)
        self.idleTimer.start(0)

        self.renderTimer = QtCore.QTimer()
        self.renderTimer.timeout.connect(self.renderTimerEvent)
        self.renderTimer.start(25)

        self.cam0Event()

        self.timerStart = time.time()

    def setTrackball(self, _trackball):
        self.trackball = _trackball
        self.cam0Event()

    def initUI(self):
        self.setGeometry(0, 0, 1280, 720)
        # self.setWindowTitle('Toolbar')

        self.glwidget = GLWidget(self)
        self.glwidget.setGeometry(0, 30, 1280, 720)
        self.glwidget.sim = self.sim
        self.glwidget.renderCallback = self.render

    def createAction(self, name, handler=None):
        action = QtWidgets.QAction(name, self)
        if handler is not None:
            action.triggered.connect(handler)
        return action

    def initActions(self):
        # Create actions
        self.resetAction = QtWidgets.QAction('Reset', self)
        self.resetAction.triggered.connect(self.resetEvent)

        self.playAction = QtWidgets.QAction('Play', self)
        self.playAction.setCheckable(True)
        self.playAction.setShortcut('Space')

        self.animAction = QtWidgets.QAction('Anim', self)
        self.animAction.setCheckable(True)

        self.captureAction = QtWidgets.QAction('Capture', self)
        self.captureAction.setCheckable(True)

        self.movieAction = QtWidgets.QAction('Movie', self)
        self.movieAction.triggered.connect(self.movieEvent)

        self.screenshotAction = QtWidgets.QAction('Screenshot', self)
        self.screenshotAction.triggered.connect(self.screenshotEvent)

        # Camera Menu
        self.cam0Action = QtWidgets.QAction('Camera0', self)
        self.cam0Action.triggered.connect(self.cam0Event)

        self.cam1Action = QtWidgets.QAction('Camera1', self)
        self.cam1Action.triggered.connect(self.cam1Event)

        self.printCamAction = QtWidgets.QAction('Print Camera', self)
        self.printCamAction.triggered.connect(self.printCamEvent)

    def initToolbarActions(self):
        self.toolbar_actions = []
        self.toolbar_actions.append(self.resetAction)
        self.toolbar_actions.append(self.playAction)
        self.toolbar_actions.append(self.animAction)
        self.toolbar_actions.append(None)
        self.toolbar_actions.append(self.screenshotAction)
        self.toolbar_actions.append(self.captureAction)
        self.toolbar_actions.append(self.movieAction)

    def initToolbar(self):
        """ Read self.toolbar_actions and add them to the toolbar.
        None will be interpreted as separator """
        # Create a toolbar
        self.toolbar = self.addToolBar('Control')
        for action in self.toolbar_actions:
            if action is None:
                self.toolbar.addSeparator()
            elif isinstance(action, QtWidgets.QAction):
                self.toolbar.addAction(action)
            elif isinstance(action, QtWidgets.QWidget):
                self.toolbar.addWidget(action)

        self.rangeSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.rangeSlider.valueChanged[int].connect(self.rangeSliderEvent)
        self.toolbar.addWidget(self.rangeSlider)

    def initMenu(self):
        menubar = self.menuBar()
        self.fileMenu = menubar.addMenu('&File')

        # Camera menu
        self.cameraMenu = menubar.addMenu('&Camera')
        self.cameraMenu.addAction(self.cam0Action)
        self.cameraMenu.addAction(self.cam1Action)
        self.cameraMenu.addSeparator()
        self.cameraMenu.addAction(self.printCamAction)

        # Recording menu
        self.recordingMenu = menubar.addMenu('&Recording')
        self.recordingMenu.addAction(self.screenshotAction)
        self.recordingMenu.addSeparator()
        self.recordingMenu.addAction(self.captureAction)
        self.recordingMenu.addAction(self.movieAction)
        self.menuBar = menubar

    def idleTimerEvent(self):
        deltaTime = time.time() - self.timerStart
        doCapture = False
        # Do animation
        if self.animAction.isChecked():
            for i in range(10):
                v = self.rangeSlider.value() + 1
                if v <= self.rangeSlider.maximum():
                    self.rangeSlider.setValue(v)
                else:
                    self.animAction.setChecked(False)
                if v % 100 == 1:
                    doCapture = True
        # Do play
        elif self.playAction.isChecked():
            num_iter = int(deltaTime/self.sim.dt+0.5)
            print deltaTime, self.sim.dt, num_iter
            for i in range(num_iter):
                if self._step_callback is not None:
                    self._step_callback(self.sim)
                result = self.sim.step()
                if result:
                    self.playAction.setChecked(False)
                if doCapture==False:
                    doCapture = (self.sim.num_frames() % 4 == 1)

        if self.captureAction.isChecked() and doCapture:
            self.glwidget.capture('capture')

        self.timerStart = time.time()

    def renderTimerEvent(self):
        self.glwidget.updateGL()
        if hasattr(self.sim, 'statusMessage'):
            message = self.sim.statusMessage()
        else:
            message = '# frames = %d' % self.sim.num_frames()
        self.statusBar().showMessage(message)
        self.rangeSlider.setRange(0, self.sim.num_frames() - 1)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            print 'Escape key pressed! Bye.'
            self.close()
        if 0 <= event.key() and event.key() < 256:  # If key is ascii
            key = chr(event.key())
            if self._keyboard_callback is not None:
                self._keyboard_callback(self.sim, key)
            if hasattr(self.sim, 'key_pressed'):
                self.sim.key_pressed(key)

    def rangeSliderEvent(self, value):
        self.sim.set_frame(value)

    def screenshotEvent(self):
        self.glwidget.capture()

    def movieEvent(self):
        cmd = 'avconv -r 200 -i ./captures/capture.%04d.png output.mp4'
        print 'Movie command:', cmd
        os.system(cmd)

    def resetEvent(self):
        self.sim.reset()

    def cam0Event(self):
        if self.trackball is not None:
            self.glwidget.tb = eval(repr(self.trackball))  # Make a deepcopy
            return
        self.glwidget.tb = Trackball(phi=-0.0, theta=0.0, zoom=1.0,
                                     rot=[-0.02, -0.71, -0.02, 0.71],
                                     trans=[0.02, 0.09, -0.69])

    def cam1Event(self):
        self.glwidget.tb = Trackball(phi=-0.0, theta=0.0, zoom=1.0,
                                     rot=[-0.02, -0.71, -0.02, 0.71],
                                     trans=[0.06, 0.26, -1.23])

    def printCamEvent(self):
        print 'printCamEvent'
        print '----'
        print repr(self.glwidget.tb)
        print '----'


def run(title='QT Window', simulation=None, trackball=None,
        step_callback=None, keyboard_callback=None, render_callback=None, cls=PyDartQtWindow):
    glutInit(())
    app = QtWidgets.QApplication([title])
    w = cls(title, simulation, render_callback)
    w.setTrackball(trackball)
    w._step_callback = step_callback
    w._keyboard_callback = keyboard_callback
    w.show()
    app.exec_()


if __name__ == '__main__':
    run()
