/**
 * React Native App with Ghost Guide - Bicep Curl Playback Demo
 * Shows live skeleton from pose-landmarks and ghost skeleton from bicep curl reference
 */

import {
  StatusBar,
  StyleSheet,
  useColorScheme,
  View,
  Text,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import Slider from '@react-native-community/slider';
import {
  SafeAreaProvider,
  SafeAreaView,
  useSafeAreaInsets,
} from 'react-native-safe-area-context';
import {
  GhostGuideCore,
  ProcessResult,
} from 'react-native-ghost-guide';
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { PoseLandmarks } from 'react-native-pose-landmarks';
import Svg, { Circle, Line } from 'react-native-svg';
import bicepCurlFrames from './assets/bicep_curl_frames.json';
import shoulderPressFrames from './assets/shoulder_press_frames.json';
import frontRaiseFrames from './assets/front_raise_frames.json';
import lateralRaiseFrames from './assets/lateral_raise_frames.json';
import tricepsExtensionFrames from './assets/triceps_extension_frames.json';

const EXERCISES = {
  bicep_curl: {
    label: 'Bicep Curl',
    frames: bicepCurlFrames,
  },
  shoulder_press: {
    label: 'Shoulder Press',
    frames: shoulderPressFrames,
  },
  front_raise: {
    label: 'Front Raise',
    frames: frontRaiseFrames,
  },
  lateral_raise: {
    label: 'Lateral Raise',
    frames: lateralRaiseFrames,
  },
  triceps_extension: {
    label: 'Triceps Extension',
    frames: tricepsExtensionFrames,
  },
} as const;

function AppContent() {
  const isDarkMode = useColorScheme() === 'dark';
  const insets = useSafeAreaInsets();
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [repCount, setRepCount] = useState(0);
  const [checkpoint, setCheckpoint] = useState(0);
  const [isAligned, setIsAligned] = useState(false);
  const [status, setStatus] = useState('Initializing...');
  const [exerciseKey, setExerciseKey] = useState<keyof typeof EXERCISES>(
    'bicep_curl'
  );
  const [showExerciseDropdown, setShowExerciseDropdown] = useState(false);

  const exerciseConfig = useMemo(() => EXERCISES[exerciseKey], [exerciseKey]);
  const totalFrames = exerciseConfig.frames.length;

  // Playback states
  const [isPlaying, setIsPlaying] = useState(false);
  const applyGhostPoseRef = useRef(false);
  const [currentFrameIndex, _setCurrentFrameIndex] = useState(0);
  const currentFrameRef = useRef(currentFrameIndex);
  const setCurrentFrameIndex = useCallback((value: any) => {
    const newVal =
      typeof value === 'function' ? value(currentFrameRef.current) : value;
    currentFrameRef.current = newVal;
    _setCurrentFrameIndex(newVal);
  }, []);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const playbackInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // Live skeleton from pose landmarks
  const landmarksBufferRef = useRef<number[]>([]);

  useEffect(() => {
    // Load reference frames to GhostGuide
    try {
      const reference = GhostGuideCore.createReferenceFromFrames(
        exerciseConfig.frames,
        exerciseKey
      );
      GhostGuideCore.loadReference(reference);
      console.log('Reference loaded with frames:', totalFrames);
      setStatus(`Reference loaded: ${totalFrames} frames`);
      currentFrameRef.current = 0;
      _setCurrentFrameIndex(0);
    } catch (e) {
      setStatus(`Error: ${e}`);
    }
  }, [exerciseConfig.frames, exerciseKey, totalFrames]);

  useEffect(() => {
    // Initialize pose landmarks
    const initialized = PoseLandmarks.initPoseLandmarker();
    if (initialized) {
      setStatus('Pose detection initialized - stand in front of camera');
    }

    // Start pose detection interval
    const interval = setInterval(() => {
      const buffer = PoseLandmarks.getLandmarksBuffer();
      if (buffer.length === 33 * 4) {
        landmarksBufferRef.current = buffer;
        try {
          const res = GhostGuideCore.processLandmarksBufferWithReference(
            buffer,
            exerciseConfig.frames,
            {
              applyReferencePose: applyGhostPoseRef.current,
              frameIndex: currentFrameRef.current,
            }
          );
          if (!res) return;
          setResult(res);
          setRepCount(res.repCount);
          setCheckpoint(res.currentCheckpointIndex);
          setIsAligned(res.isAligned);
        } catch (e) {
          console.error('Process frame error:', e);
        }
      }
    }, 100);

    return () => {
      clearInterval(interval);
      PoseLandmarks.closePoseLandmarker();
    };
  }, [exerciseConfig.frames]);

  const startPlayback = () => {
    if (playbackInterval.current) return;

    setIsPlaying(true);
    applyGhostPoseRef.current = true;

    playbackInterval.current = setInterval(() => {
      const next = currentFrameRef.current + 1;
      if (next >= totalFrames) {
        // Loop back to beginning
        currentFrameRef.current = 0;
        _setCurrentFrameIndex(0);
      } else {
        currentFrameRef.current = next;
        _setCurrentFrameIndex(next);
      }
    }, (1000 / 30) / playbackSpeed);
  };

  const stopPlayback = () => {
    if (playbackInterval.current) {
      clearInterval(playbackInterval.current);
      playbackInterval.current = null;
    }
    setIsPlaying(false);
  };

  const pausePlayback = () => {
    stopPlayback();
  };

  const resetPlayback = () => {
    stopPlayback();
    currentFrameRef.current = 0;
    _setCurrentFrameIndex(0);
    applyGhostPoseRef.current = false;
  };

  useEffect(() => {
    return () => {
      if (playbackInterval.current) {
        clearInterval(playbackInterval.current);
      }
    };
  }, []);

  // Background color based on theme
  const containerBgColor = isDarkMode ? '#000' : '#f5f5f5';

  const ghostPointsForRender = result?.ghostSkeleton?.points ?? null;
  const ghostUpperBodyIndices = new Set([
    11, 12, 13, 14, 15, 16, 23, 24,
  ]);


  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: containerBgColor }]}
    >
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      <View style={[styles.mainContainer, { paddingTop: insets.top }]}>

        {/* Skeleton View - Full Screen */}
        <View style={styles.skeletonContainer}>
          <Svg width="100%" height="100%" viewBox="0 0 1 1">
             {/* Live skeleton from pose landmarks (cyan/yellow) - render FIRST so ghost appears on top */}
              {landmarksBufferRef.current.length === 33 * 4 && (
               <>
                 {/* Draw skeleton connections */}
                 {[
                   [11,13],[13,15],[12,14],[14,16],
                   [11,12],[23,24],
                   [11,23],[12,24],
                 ].map(([i, j], idx) => {
                    const x1 = landmarksBufferRef.current[i * 4];
                    const y1 = landmarksBufferRef.current[i * 4 + 1];
                    const x2 = landmarksBufferRef.current[j * 4];
                    const y2 = landmarksBufferRef.current[j * 4 + 1];
                    const vis1 = landmarksBufferRef.current[i * 4 + 3];
                    const vis2 = landmarksBufferRef.current[j * 4 + 3];
                   if (vis1 < 0.5 || vis2 < 0.5) return null;
                   return (
                     <Line
                       key={`live-conn-${idx}`}
                       x1={x1}
                       y1={y1}
                       x2={x2}
                       y2={y2}
                       stroke={isAligned ? "lime" : "cyan"}
                       strokeWidth="0.005"
                     />
                   );
                 })}
                 {/* Draw landmarks */}
                 {Array.from({ length: 33 }, (_, i) => {
                    const x = landmarksBufferRef.current[i * 4];
                    const y = landmarksBufferRef.current[i * 4 + 1];
                    const vis = landmarksBufferRef.current[i * 4 + 3];
                   if (vis < 0.5) return null;
                   return (
                     <Circle
                       key={`live-lm-${i}`}
                       cx={x}
                       cy={y}
                       r="0.01"
                       fill={i >= 11 && i <= 16 ? "cyan" : "yellow"}
                     />
                   );
                 })}
               </>
             )}

             {/* Ghost/Reference skeleton from library (already calibrated) - render LAST to appear on top */}
              {ghostPointsForRender && (
                <>
                  {[
                    [11, 13],
                    [13, 15],
                    [12, 14],
                    [14, 16],
                    [11, 12],
                    [11, 23],
                    [12, 24],
                    [23, 24],
                  ].map(([i, j], idx) => {
                    const p1 = ghostPointsForRender[i];
                    const p2 = ghostPointsForRender[j];
                    if (!ghostUpperBodyIndices.has(i) || !ghostUpperBodyIndices.has(j)) {
                      return null;
                    }
                    if (!p1 || !p2) return null;
                    return (
                      <Line
                        key={`ghost-conn-${idx}`}
                        x1={p1.x}
                        y1={p1.y}
                        x2={p2.x}
                        y2={p2.y}
                        stroke="rgba(255, 0, 0, 0.95)"
                        strokeWidth="0.012"
                      />
                    );
                  })}
                  {ghostPointsForRender
                    .map((p: any, i: number) => ({ p, i }))
                    .filter(({ i }) => ghostUpperBodyIndices.has(i))
                    .map(({ p, i }) => (
                      <Circle
                        key={`ghost-lm-${i}`}
                        cx={p.x}
                        cy={p.y}
                        r="0.02"
                        fill="rgba(255, 0, 0, 0.9)"
                      />
                    ))}
               </>
             )}
           </Svg>
        </View>

        <ScrollView
          style={styles.infoOverlay}
          contentContainerStyle={styles.infoContent}
        >
            <Text style={styles.title}>Ghost Guide - {exerciseConfig.label}</Text>
          <Text style={styles.status}>{status}</Text>

        <View style={styles.debugBox}>
          <Text style={styles.debugText}>Ghost: {ghostPointsForRender ? 'YES' : 'NO'}</Text>
        </View>
          
          <View style={styles.resultBox}>
            <Text style={styles.resultText}>Reps: {repCount}</Text>
            <Text style={styles.resultText}>Checkpoint: {checkpoint}</Text>
            <Text style={[styles.resultText, isAligned && styles.aligned]}>
              Aligned: {isAligned ? 'YES' : 'NO'}
            </Text>
            <Text style={styles.resultText}>
              Frame: {currentFrameRef.current + 1} / {totalFrames}
            </Text>
          </View>

          {/* Playback Controls */}
          <View style={styles.controlsBox}>
<View style={styles.dropdownContainer}>
            <TouchableOpacity
              style={styles.dropdownTrigger}
              onPress={() => setShowExerciseDropdown(!showExerciseDropdown)}
            >
              <Text style={styles.dropdownTriggerText}>{exerciseConfig.label}</Text>
              <Text style={styles.dropdownArrow}>{showExerciseDropdown ? '▲' : '▼'}</Text>
            </TouchableOpacity>
            {showExerciseDropdown && (
              <View style={styles.dropdownList}>
                {Object.entries(EXERCISES).map(([key, config]) => (
                  <TouchableOpacity
                    key={key}
                    style={[
                      styles.dropdownOption,
                      exerciseKey === key && styles.dropdownOptionActive,
                    ]}
                    onPress={() => {
                      setExerciseKey(key as keyof typeof EXERCISES)
                      setShowExerciseDropdown(false)
                    }}
                  >
                    <Text style={[
                      styles.dropdownOptionText,
                      exerciseKey === key && styles.dropdownOptionTextActive,
                    ]}>
                      {config.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            )}
          </View>
            <View style={styles.buttonRow}>
              <TouchableOpacity
                style={styles.button}
                onPress={resetPlayback}
              >
                <Text style={styles.buttonText}>Reset</Text>
              </TouchableOpacity>

              {!isPlaying ? (
                <TouchableOpacity
                  style={[styles.button, styles.playButton]}
                  onPress={startPlayback}
                >
                  <Text style={styles.buttonText}>Play</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity
                  style={[styles.button, styles.pauseButton]}
                  onPress={stopPlayback}
                >
                  <Text style={styles.buttonText}>Stop</Text>
                </TouchableOpacity>
              )}

              <TouchableOpacity
                style={styles.button}
                onPress={() =>
                  setPlaybackSpeed(playbackSpeed === 1 ? 2 : playbackSpeed === 2 ? 0.5 : 1)
                }
              >
                <Text style={styles.buttonText}>{playbackSpeed}x</Text>
              </TouchableOpacity>
            </View>

            {/* Frame Slider */}
            <View style={styles.sliderContainer}>
              <Text style={styles.sliderLabel}>Frame</Text>
              <Slider
                style={styles.slider}
                minimumValue={0}
                maximumValue={Math.max(0, totalFrames - 1)}
                value={currentFrameRef.current}
                onValueChange={(value) => {
                  stopPlayback();
                  const next = Math.round(value);
                  setCurrentFrameIndex(next);
                }}
                step={1}
                minimumTrackTintColor="#007AFF"
                maximumTrackTintColor="#ddd"
              />
            </View>
          </View>
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  mainContainer: {
    flex: 1,
    position: 'relative',
  },
  skeletonContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  infoOverlay: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    maxHeight: '45%',
    padding: 20,
  },
  infoContent: {
    paddingBottom: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#fff',
  },
  status: {
    fontSize: 14,
    color: '#aaa',
    marginBottom: 20,
  },
  debugText: {
    fontSize: 12,
    color: '#ff0',
    marginBottom: 5,
    fontFamily: 'monospace',
  },
  debugBox: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 8,
    borderRadius: 4,
    marginBottom: 10,
  },
  resultBox: {
    backgroundColor: 'rgba(255,255,255,0.9)',
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
  },
  resultText: {
    fontSize: 16,
    marginBottom: 5,
  },
  aligned: {
    color: 'green',
    fontWeight: 'bold',
  },
  controlsBox: {
    backgroundColor: 'rgba(255,255,255,0.95)',
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 15,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 6,
    minWidth: 80,
    alignItems: 'center',
  },
  playButton: {
    backgroundColor: '#34C759',
  },
  pauseButton: {
    backgroundColor: '#FF9500',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  sliderContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  sliderLabel: {
    fontSize: 14,
    fontWeight: '600',
    marginRight: 10,
    width: 50,
  },
  slider: {
    flex: 1,
    height: 40,
  },
  dropdownContainer: {
    marginBottom: 15,
  },
  dropdownTrigger: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#007AFF',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 6,
  },
  dropdownTriggerText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  dropdownArrow: {
    color: '#fff',
    fontSize: 14,
    marginLeft: 8,
  },
  dropdownList: {
    backgroundColor: '#fff',
    borderRadius: 6,
    marginTop: 4,
    overflow: 'hidden',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 4,
  },
  dropdownOption: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  dropdownOptionActive: {
    backgroundColor: '#e8f0fe',
  },
  dropdownOptionText: {
    fontSize: 16,
    color: '#333',
  },
  dropdownOptionTextActive: {
    color: '#007AFF',
    fontWeight: '600',
  },
});

function App() {
  return (
    <SafeAreaProvider>
      <AppContent />
    </SafeAreaProvider>
  );
}

export default App;
