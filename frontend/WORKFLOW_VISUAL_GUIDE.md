# Project Workflow Visual Guide

## Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROJECT DETAIL PAGE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ← Back to Projects              Project Title                Edit  Del  │
│                                  Created on: Date                         │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  📅 Project Information                                             ││
│  │  Description text here...                                           ││
│  │  🎬 Video uploaded  📍 Location set  📐 Homography solved           ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                           │
│  ┌──────────────┬──────────────────────────────────────────────────────┐│
│  │   STEPPER    │              STEP CONTENT                            ││
│  │   (Left)     │              (Right)                                 ││
│  ├──────────────┼──────────────────────────────────────────────────────┤│
│  │              │                                                       ││
│  │ ① Upload     │  ┌─────────────────────────────────────────────┐   ││
│  │   Video      │  │                                              │   ││
│  │   ✓ Done     │  │         Active Step Content                 │   ││
│  │              │  │         (Transitions smoothly)               │   ││
│  │ ② Capture    │  │                                              │   ││
│  │   Frame      │  │  • VideoUpload component                     │   ││
│  │   ⚠ Pending  │  │  • LocationPicker component                  │   ││
│  │              │  │  • HomographyPicker component                │   ││
│  │ ③ Set        │  │  • ProcessingPanel component                 │   ││
│  │   Location   │  │                                              │   ││
│  │   ✓ Done     │  │                                              │   ││
│  │              │  └─────────────────────────────────────────────┘   ││
│  │ ④ Homography │                                                      ││
│  │   ⚠ Pending  │  ┌─────────────────────────────────────────────┐   ││
│  │              │  │  ← Back    Step 1 of 5    Next →            │   ││
│  │ ⑤ Review     │  └─────────────────────────────────────────────┘   ││
│  │   & Run      │                                                      ││
│  │   ⚠ Pending  │                                                      ││
│  │              │                                                      ││
│  └──────────────┴──────────────────────────────────────────────────────┘│
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Mobile Layout (< 768px)

```
┌────────────────────────────────┐
│     PROJECT DETAIL PAGE        │
├────────────────────────────────┤
│                                │
│  ← Back                        │
│  Project Title                 │
│  Created on: Date              │
│  Edit  Del                     │
│                                │
│  ┌──────────────────────────┐ │
│  │  📅 Project Info         │ │
│  │  🎬 📍 📐                │ │
│  └──────────────────────────┘ │
│                                │
│  ┌──────────────────────────┐ │
│  │     STEPPER (Stacked)    │ │
│  ├──────────────────────────┤ │
│  │ ① Upload Video  ✓        │ │
│  │ ② Capture Frame ⚠        │ │
│  │ ③ Set Location  ✓        │ │
│  │ ④ Homography    ⚠        │ │
│  │ ⑤ Review & Run  ⚠        │ │
│  └──────────────────────────┘ │
│                                │
│  ┌──────────────────────────┐ │
│  │                          │ │
│  │    Active Step Content   │ │
│  │    (Full Width)          │ │
│  │                          │ │
│  └──────────────────────────┘ │
│                                │
│  ┌──────────────────────────┐ │
│  │  ← Back    Next →        │ │
│  └──────────────────────────┘ │
│                                │
└────────────────────────────────┘
```

## Step Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW PROGRESSION                         │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Upload Video
├─ User uploads video file
├─ System processes video
├─ Frame extraction triggered
└─ Status: ✓ Complete → Next enabled

Step 2: Capture Key Frame
├─ System extracts first frame
├─ Screenshot saved as media asset
├─ Frame displayed in UI
└─ Status: ✓ Complete → Next enabled

Step 3: Set Location
├─ User searches address OR
├─ User clicks map OR
├─ User enters coordinates
├─ Location saved to project
└─ Status: ✓ Complete → Next enabled

Step 4: Configure Homography
├─ User picks 4+ point pairs
├─ System validates pairs
├─ User clicks "Solve"
├─ Homography matrix computed
└─ Status: ✓ Complete → Next enabled

Step 5: Review & Run
├─ System checks all prerequisites
├─ User reviews summary
├─ User clicks "Run Analysis"
└─ Processing begins
```

## Status Badge System

```
┌─────────────────────────────────────────────────────────────┐
│                      BADGE TYPES                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✓ Done     │ Green  │ Step completed successfully         │
│  ⚠ Pending  │ Yellow │ Step not started or in progress     │
│  ✗ Error    │ Red    │ Step has blocking error             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Step Content Examples

### Step 1: Upload Video

```
┌──────────────────────────────────────────────────────────────┐
│  Upload Video                                                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Upload your CCTV or dashcam video for accident analysis.    │
│  The video will be processed to extract a key frame.         │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ✓ Video uploaded successfully                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                         │ │
│  │         📁 Drag video file here or click               │ │
│  │            Supports MP4, AVI, MOV (max 100MB)          │ │
│  │                                                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Step 2: Capture Key Frame

```
┌──────────────────────────────────────────────────────────────┐
│  Capture Key Frame                                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  A key frame from your video is needed for homography        │
│  configuration. This frame will be used to map coordinates.  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ✓ Key frame captured successfully.                    │ │
│  │    You can proceed to set the location.                │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Step 3: Set Location

```
┌──────────────────────────────────────────────────────────────┐
│  Set Location                                                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Specify the geographic location where the video was         │
│  recorded. This helps with accurate speed calculations.      │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ✓ Location set: 800 140th Ave NE, Bellevue, WA       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  Address: [Search for an address...          ] [Search]     │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                         │ │
│  │                    🗺️ Google Map                       │ │
│  │                   (Interactive)                         │ │
│  │                                                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  Latitude:  [47.616912]    Longitude: [-122.143269]         │
│                                                               │
│  [Set Location]                                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Step 4: Configure Homography

```
┌──────────────────────────────────────────────────────────────┐
│  Configure Homography                                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Map points from your video frame to real-world map          │
│  coordinates. This enables accurate speed calculations.      │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ✓ Homography solved successfully!                     │ │
│  │    You can now proceed to run processing.              │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  [Pick Points] [New Pair]                                    │
│                                                               │
│  ┌──────────────────────┬──────────────────────────────────┐│
│  │   CCTV Image (A)     │   Google Map (B)                ││
│  ├──────────────────────┼──────────────────────────────────┤│
│  │                      │                                  ││
│  │   [Screenshot]       │   [Interactive Map]             ││
│  │   • Point markers    │   • Point markers               ││
│  │                      │                                  ││
│  └──────────────────────┴──────────────────────────────────┘│
│                                                               │
│  Point Pairs (4) ✓ Ready                                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  #  │  A (Normalized)  │  B (Lat, Lng)  │  Actions    │ │
│  │  1  │  (0.2500, 0.3000)│  (47.6, -122.1)│  🗑️         │ │
│  │  2  │  (0.7500, 0.3000)│  (47.6, -122.2)│  🗑️         │ │
│  │  3  │  (0.2500, 0.7000)│  (47.7, -122.1)│  🗑️         │ │
│  │  4  │  (0.7500, 0.7000)│  (47.7, -122.2)│  🗑️         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  [✓ Solve Homography]                                        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Step 5: Review & Run

```
┌──────────────────────────────────────────────────────────────┐
│  Review & Run Processing                                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Review your project configuration and start video           │
│  processing to detect vehicles and calculate speeds.         │
│                                                               │
│  Prerequisites Summary                                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  🎬 Video Upload          ✓ Complete                   │ │
│  │  📸 Key Frame             ✓ Complete                   │ │
│  │  📍 Location              ✓ Complete                   │ │
│  │  📐 Homography            ✓ Complete                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ✓ All prerequisites met!                              │ │
│  │    You can now run video processing.                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Video Processing                                       │ │
│  │  Run YOLO detection and ByteTrack tracking             │ │
│  │                                                         │ │
│  │  [▶ Run Analysis]                                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  Processing Runs                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Status    │  Started  │  Duration  │  Artifacts      │ │
│  │  Running   │  10:30 AM │  Running   │  🎬 📄          │ │
│  │  Completed │  09:15 AM │  45s       │  🎬 📄          │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Transition Animations

### Forward (Next Button)

```
Step 1                Step 2
┌─────────┐          ┌─────────┐
│ Content │  ─────>  │ Content │
│    A    │  slide   │    B    │
└─────────┘  left    └─────────┘
```

### Backward (Back Button)

```
Step 2                Step 1
┌─────────┐          ┌─────────┐
│ Content │  <─────  │ Content │
│    B    │  slide   │    A    │
└─────────┘  right   └─────────┘
```

## Color Scheme

```
┌─────────────────────────────────────────────────────────────┐
│                      COLOR PALETTE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Success (Green)  │ #40c057 │ Completed steps, success     │
│  Warning (Yellow) │ #fab005 │ Pending steps, warnings      │
│  Error (Red)      │ #fa5252 │ Failed steps, errors         │
│  Primary (Blue)   │ #228be6 │ Active step, primary actions │
│  Gray (Dimmed)    │ #868e96 │ Disabled, secondary text     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Responsive Breakpoints

```
┌─────────────────────────────────────────────────────────────┐
│                    RESPONSIVE LAYOUT                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Desktop   │  > 1200px  │  Two columns (3/9 split)         │
│  Tablet    │  768-1200  │  Two columns (3/9 split)         │
│  Mobile    │  < 768px   │  Single column (stacked)         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## User Interaction Flow

```
1. User lands on project detail page
   ↓
2. Sees workflow stepper with current status
   ↓
3. Clicks on incomplete step OR clicks Next
   ↓
4. Views step content with guidance
   ↓
5. Completes required action
   ↓
6. System updates step status
   ↓
7. User proceeds to next step
   ↓
8. Repeats until all steps complete
   ↓
9. Reviews summary in Step 5
   ↓
10. Clicks "Run Analysis"
    ↓
11. Processing begins
```

## Key Features Visualization

### Sticky Stepper (Desktop)

```
┌──────────────┬────────────────────┐
│   STEPPER    │   CONTENT          │
│   (Sticky)   │   (Scrollable)     │
│              │                    │
│ ① Step 1     │ ┌────────────────┐│
│ ② Step 2     │ │                ││
│ ③ Step 3     │ │   Long         ││
│ ④ Step 4     │ │   Content      ││
│ ⑤ Step 5     │ │   Scrolls      ││
│              │ │                ││
│              │ │                ││
│              │ │                ││
│              │ └────────────────┘│
└──────────────┴────────────────────┘
     ↑ Stays visible while scrolling
```

### Direct Step Access

```
User can click any step at any time:

① ──> Click ──> Jump to Step 1
②              (No restrictions)
③ ──> Click ──> Jump to Step 3
④              (Flexible navigation)
⑤ ──> Click ──> Jump to Step 5
```

### Validation Feedback

```
┌─────────────────────────────────────┐
│  Step Status                         │
├─────────────────────────────────────┤
│                                      │
│  Complete:  ✓ Done (Green badge)    │
│  Pending:   ⚠ Pending (Yellow)      │
│  Error:     ✗ Error (Red)           │
│                                      │
│  Hover badge → Shows tooltip        │
│  with detailed message              │
│                                      │
└─────────────────────────────────────┘
```

This visual guide provides a comprehensive overview of the workflow UI structure, layout, and user interactions.
