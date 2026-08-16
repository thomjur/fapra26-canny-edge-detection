# CUDA Canny Edge Detection

Dieses Projekt implementiert einen Canny-Kantendetektor mit CUDA. Die Pipeline
besteht aus:

1. Gaussian Blur
2. Sobel-Filter
3. Non-Maximum Suppression
4. Hysterese-Schwellwertverfahren

## Voraussetzungen

- NVIDIA-GPU
- CUDA Toolkit mit `nvcc`
- C++20-Unterstützung

Die Build-Skripte verwenden standardmäßig die CUDA-Architektur `sm_75`. Bei
Bedarf kann `ARCH` im jeweiligen Skript angepasst werden.

## Kompilieren und Ausführen

Im Projektverzeichnis:

```bash
./build_release.sh
./run_release.sh
```

Für eine Debug-Version:

```bash
./build_debug.sh
./run_debug.sh
```

Die Run-Skripte verwenden das Beispielbild aus `assets/`. Das kompilierte
Programm kann auch direkt mit eigenen Dateien gestartet werden:

```bash
./bin/canny_release.out eingabe.jpg ausgabe.png
```

## Ausgaben

Neben der angegebenen Ergebnisdatei werden Zwischenbilder in `assets/`
gespeichert:

- `gaussian.png`
- `sobel_grad.png`
- `sobel_dir.png`
- `nms.png`
- `hysteresis.png`
- `hysteresis_fused.png` bei aktivierter Fused Pipeline

## Build-Optionen

Die Optionen werden als Umgebungsvariablen beim Kompilieren übergeben. Der
jeweilige Wert wird fest in das Programm einkompiliert.

### Pipeline-Varianten

```bash
GAUSSIAN_OPT=1 SOBEL_OPT=5 PIPELINE_FUSED=1 ./build_release.sh
```

Wichtige Optionen:

| Variable | Bedeutung |
| --- | --- |
| `GAUSSIAN_OPT=0..2` | Gaussian: naiv, Shared Memory oder Pinned Memory |
| `SOBEL_OPT=0..5` | Sobel-Implementierung; `4` ist Bucket, `5` ist die geteilte Gx/Gy-Bucket-Variante |
| `NMS_OPT=0..2` | NMS: naiv, Shared Memory oder Pinned Memory |
| `HYSTERESIS_OPT=0..2` | Hysterese-Implementierung |
| `PIPELINE_FUSED=1` | Fused Pipeline aktiv, Standardwert |
| `PIPELINE_FUSED=0` | Fused Pipeline deaktiviert |
| `BENCHMARK_RUNS=n` | Anzahl der Messläufe, Standardwert `1` |
| `WARMUP=1` | Warm-up-Lauf aktivieren |
| `WARMUP=0` | Warm-up-Lauf deaktivieren |

Beispiel für einen Benchmark mit zehn Messläufen ohne Fused Pipeline:

```bash
SOBEL_OPT=5 PIPELINE_FUSED=0 BENCHMARK_RUNS=10 ./build_release.sh
```

Die Fused Pipeline wird zusätzlich zur normalen Einzelstufenmessung
ausgeführt, wenn sie aktiviert ist. Wird sie deaktiviert, läuft nur die
normale Pipeline.

## Projektstruktur

- `src/`: CUDA- und C++-Quellcode
- `assets/`: Eingabe-, Ergebnis- und Zwischenbilder
- `build_release.sh`: optimierter Release-Build
- `build_debug.sh`: Debug-Build
- `run_release.sh`: Release-Programm starten
- `run_debug.sh`: Debug-Programm starten
