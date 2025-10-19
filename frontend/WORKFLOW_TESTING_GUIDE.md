# Project Workflow Testing Guide

This document provides a comprehensive testing checklist for the new project workflow UI implemented in `projects.$projectId.tsx`.

## Overview

The project detail page has been refactored from a tabbed layout to a guided workflow with a stepper UI. The workflow consists of 5 steps that guide users through video processing prerequisites.

## Components

- **ProjectWorkflow**: Main workflow component with stepper and step content
- **Route**: `app/routes/projects.$projectId.tsx`
- **Supporting Components**: VideoUpload, LocationPicker, HomographyPicker, ProcessingPanel

## Testing Checklist

### 1. Step Navigation

#### Forward Navigation

- [ ] Click "Next" button to advance from Step 1 to Step 2
- [ ] Click "Next" button to advance through all steps sequentially
- [ ] Verify "Next" button is disabled when prerequisites are not met
- [ ] Verify smooth slide-left transition when moving forward

#### Backward Navigation

- [ ] Click "Back" button to return to previous step
- [ ] Verify "Back" button is disabled on Step 1
- [ ] Verify smooth slide-right transition when moving backward

#### Direct Step Access

- [ ] Click on Step 2 in the stepper while on Step 1
- [ ] Click on Step 5 in the stepper while on Step 1
- [ ] Click on Step 1 in the stepper while on Step 5
- [ ] Verify all steps are clickable and accessible at any time

### 2. Step Status Indicators

#### Step 1: Upload Video

- [ ] Badge shows "Pending" when no video uploaded
- [ ] Badge shows "Done" when video is uploaded successfully
- [ ] Badge shows "Error" when video processing fails
- [ ] Tooltip displays error message on hover

#### Step 2: Capture Key Frame

- [ ] Badge shows "Pending" when no frame extracted
- [ ] Badge shows "Done" when frame is captured
- [ ] Badge shows "Error" when video is not uploaded
- [ ] Warning message displays during frame extraction

#### Step 3: Set Location

- [ ] Badge shows "Pending" when no location set
- [ ] Badge shows "Done" when location is configured
- [ ] Location coordinates display correctly

#### Step 4: Configure Homography

- [ ] Badge shows "Pending" when not configured
- [ ] Badge shows "Done" when homography is solved
- [ ] Badge shows "Error" when prerequisites missing
- [ ] Warning displays when in draft state

#### Step 5: Review & Run

- [ ] Badge shows "Pending" when prerequisites incomplete
- [ ] Badge shows "Done" when all prerequisites met
- [ ] Prerequisites summary displays correct status for each item

### 3. Step Content

#### Step 1: Upload Video

- [ ] VideoUpload component renders correctly
- [ ] Drag-and-drop functionality works
- [ ] File selection dialog opens on click
- [ ] Upload progress displays during upload
- [ ] Success alert shows after upload
- [ ] Error alert shows on upload failure
- [ ] Page refreshes after successful upload

#### Step 2: Capture Key Frame

- [ ] Success alert shows when frame exists
- [ ] Processing alert shows during extraction
- [ ] Error alert shows when video missing
- [ ] Warning alert shows when extraction pending

#### Step 3: Set Location

- [ ] LocationPicker component renders correctly
- [ ] Google Maps displays properly
- [ ] Address search works
- [ ] Map click sets coordinates
- [ ] Manual coordinate entry works
- [ ] Success alert shows when location set
- [ ] Page refreshes after location update

#### Step 4: Configure Homography

- [ ] HomographyPicker component renders correctly
- [ ] Screenshot displays if available
- [ ] Point picking mode works
- [ ] Pairs table displays correctly
- [ ] Solve button appears when 4+ pairs exist
- [ ] Success alert shows when solved
- [ ] Error alerts show for missing prerequisites

#### Step 5: Review & Run

- [ ] Prerequisites summary displays all 4 items
- [ ] Each prerequisite shows correct status badge
- [ ] Success alert shows when all complete
- [ ] Warning alert shows when incomplete
- [ ] ProcessingPanel renders correctly
- [ ] "Run Analysis" button is enabled when ready
- [ ] "Run Analysis" button is disabled when not ready

### 4. Data State Testing

#### Empty Project (No Data)

- [ ] Step 1 shows "No video uploaded" warning
- [ ] Step 2 shows "Upload video first" error
- [ ] Step 3 shows "Location not set" warning
- [ ] Step 4 shows "Capture key frame first" error
- [ ] Step 5 shows all prerequisites incomplete
- [ ] Navigation allows moving forward despite errors

#### Video Uploaded Only

- [ ] Step 1 shows completed status
- [ ] Step 2 shows processing or pending status
- [ ] Step 3 shows "Location not set" warning
- [ ] Step 4 shows "Set location first" error
- [ ] Step 5 shows incomplete prerequisites

#### Video + Frame Captured

- [ ] Step 1 shows completed status
- [ ] Step 2 shows completed status
- [ ] Step 3 shows "Location not set" warning
- [ ] Step 4 shows "Set location first" error
- [ ] Step 5 shows incomplete prerequisites

#### Video + Frame + Location

- [ ] Step 1 shows completed status
- [ ] Step 2 shows completed status
- [ ] Step 3 shows completed status
- [ ] Step 4 shows "Homography not configured" warning
- [ ] Step 5 shows incomplete prerequisites

#### All Prerequisites Complete

- [ ] All steps 1-4 show completed status
- [ ] Step 5 shows "All prerequisites met" success
- [ ] "Ready to Process" button is enabled
- [ ] ProcessingPanel "Run Analysis" button is enabled

#### Video Processing Error

- [ ] Step 1 shows error badge
- [ ] Error message displays in step content
- [ ] Tooltip shows error on badge hover
- [ ] User can still proceed to other steps

#### Homography Draft State

- [ ] Step 4 shows warning badge
- [ ] Warning message indicates "not solved"
- [ ] Step 5 shows prerequisites incomplete
- [ ] Solve button is available in Step 4

### 5. Responsive Layout

#### Desktop (>1200px)

- [ ] Stepper displays in left column (3 columns wide)
- [ ] Content displays in right column (9 columns wide)
- [ ] Both columns visible side-by-side
- [ ] Stepper is sticky on scroll

#### Tablet (768px - 1200px)

- [ ] Stepper displays in left column (3 columns wide)
- [ ] Content displays in right column (9 columns wide)
- [ ] Layout remains two-column
- [ ] Content is readable and functional

#### Mobile (<768px)

- [ ] Stepper stacks above content (12 columns wide)
- [ ] Content displays below stepper (12 columns wide)
- [ ] Stepper is no longer sticky
- [ ] All controls are accessible
- [ ] Touch interactions work properly

### 6. Transitions and Animations

- [ ] Slide-left transition plays when moving forward
- [ ] Slide-right transition plays when moving backward
- [ ] Transition duration is smooth (300ms)
- [ ] No content flashing or jumping
- [ ] Transitions work on all step changes

### 7. Integration with Existing Features

#### Project Header

- [ ] Project title displays correctly
- [ ] Creation date displays correctly
- [ ] Edit button opens modal
- [ ] Delete button shows confirmation
- [ ] Delete navigates to projects list
- [ ] Back button navigates to projects list

#### Project Info Card

- [ ] All status badges display correctly
- [ ] Video uploaded badge appears when video exists
- [ ] Location set badge appears when location exists
- [ ] Homography badge shows correct status
- [ ] Processing badge shows correct status

#### Data Refresh

- [ ] Video upload triggers project refresh
- [ ] Location update triggers project refresh
- [ ] Homography solve triggers project refresh
- [ ] Step statuses update after refresh
- [ ] No unnecessary page reloads

### 8. User Experience

#### Visual Feedback

- [ ] Active step is highlighted in stepper
- [ ] Completed steps show checkmark icon
- [ ] Error steps show red color
- [ ] Warning steps show yellow color
- [ ] Hover states work on all interactive elements

#### Error Handling

- [ ] Error messages are clear and actionable
- [ ] Warnings don't block progression
- [ ] Errors are displayed prominently
- [ ] Tooltips provide additional context

#### Workflow Guidance

- [ ] Step descriptions are clear
- [ ] Prerequisites are clearly stated
- [ ] Success messages confirm actions
- [ ] Next steps are obvious to users

### 9. Edge Cases

- [ ] Rapidly clicking Next/Back doesn't break state
- [ ] Clicking same step multiple times doesn't cause issues
- [ ] Browser back/forward buttons work correctly
- [ ] Page refresh preserves project state
- [ ] Concurrent updates don't cause conflicts
- [ ] Large project data renders without performance issues

### 10. Accessibility

- [ ] Keyboard navigation works (Tab, Enter)
- [ ] Screen reader announces step changes
- [ ] Focus indicators are visible
- [ ] Color contrast meets WCAG standards
- [ ] Interactive elements have proper ARIA labels

## Test Scenarios

### Scenario 1: New Project Setup

1. Create a new project
2. Navigate to project detail page
3. Verify all steps show incomplete status
4. Upload a video in Step 1
5. Wait for frame extraction in Step 2
6. Set location in Step 3
7. Configure homography in Step 4
8. Review and run processing in Step 5

### Scenario 2: Existing Project Review

1. Open a project with all prerequisites complete
2. Verify all steps show completed status
3. Navigate through steps to review configuration
4. Verify all data displays correctly
5. Run processing from Step 5

### Scenario 3: Error Recovery

1. Open a project with video processing error
2. Verify error displays in Step 1
3. Re-upload video to resolve error
4. Continue through remaining steps

### Scenario 4: Mobile Workflow

1. Open project detail on mobile device
2. Verify stepper stacks above content
3. Complete workflow using touch interactions
4. Verify all features work on mobile

## Performance Considerations

- [ ] Initial page load is fast (<2s)
- [ ] Step transitions are smooth (60fps)
- [ ] No memory leaks during navigation
- [ ] Images load progressively
- [ ] API calls are optimized (no unnecessary requests)

## Browser Compatibility

Test in the following browsers:

- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)
- [ ] Mobile Safari (iOS)
- [ ] Chrome Mobile (Android)

## Known Limitations

1. Page refreshes after video upload and location update (by design)
2. Stepper allows jumping to any step regardless of completion status
3. Homography configuration requires manual point selection

## Future Enhancements

1. Add step completion persistence in URL query params
2. Implement auto-save for draft states
3. Add undo/redo functionality
4. Improve mobile touch interactions for homography picking
5. Add keyboard shortcuts for step navigation
