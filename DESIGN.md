---
name: hplugins
description: A technical parts plate for community Heretic plugins, built from hairlines, tabular numerics, and one cobalt accent.
colors:
  selection-cobalt: "#1D4ED8"
  cool-paper: "#EDEFF1"
  plate-white: "#F7F8F8"
  graphite-ink: "#15191C"
  machine-grey: "#4A565C"
  hairline-grey: "#C6CDD1"
  night-plate: "#0D1215"
  night-ink: "#E7ECEB"
  night-hairline: "#2B373D"
  night-cobalt: "#6E8BFF"
typography:
  display:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "clamp(2.4rem, 4.6vw, 4.1rem)"
    fontWeight: 500
    lineHeight: 1.02
    letterSpacing: "-0.04em"
  body:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.98rem"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "0.72rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0.12em"
rounded:
  sm: "2px"
  pill: "999px"
spacing:
  xs: "6px"
  sm: "14px"
  md: "24px"
  lg: "48px"
components:
  button-primary:
    backgroundColor: "{colors.selection-cobalt}"
    textColor: "{colors.plate-white}"
    rounded: "{rounded.sm}"
    padding: "10px 16px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.machine-grey}"
    rounded: "{rounded.sm}"
    padding: "7px 14px"
  input-field:
    backgroundColor: "{colors.plate-white}"
    textColor: "{colors.graphite-ink}"
    rounded: "{rounded.sm}"
    padding: "7px 10px"
  index-row:
    backgroundColor: "transparent"
    textColor: "{colors.graphite-ink}"
    rounded: "{rounded.sm}"
    padding: "20px 8px"
---

# Design System: hplugins

## Overview

**Creative North Star: "The Parts Plate"**

A registry treated as an exploded engineering plate rather than a gallery of cards. Every entry is a numbered part on a ruled sheet. A fixed index spine holds the totals; the main column carries the parts list and, on an entry page, the full sheet. The page behaves like a document you consult, not a feed you scroll.

The atmosphere is cool, dry, and legible. Hairlines do the work that shadows would do elsewhere. Blue appears once, on the active selection and on actions, so it reads as a decision rather than decoration. Both themes share the same structure: cool paper by day, night plate after dark.

**Key Characteristics:**

- Two-column plate with a sticky index spine and a wide reading column
- Numbered list rows with tabular mono indices instead of uniform cards
- One accent, reserved for active state and controls
- Status carried by a rotated square stamp plus text, never color alone

## Colors

One accent on a neutral plate. Dark mode swaps the neutral ramp and lightens the accent for contrast; accent placement stays identical.

### Primary
- **Selection Cobalt** (#1D4ED8): Active filter, primary buttons, source-link arrows. Night mode uses Night Cobalt (#6E8BFF).

### Neutral
- **Cool Paper** (#EDEFF1): Page ground in light mode.
- **Plate White** (#F7F8F8): Spine surface and input backgrounds.
- **Graphite Ink** (#15191C): Headings and body copy.
- **Machine Grey** (#4A565C): Secondary copy and labels.
- **Hairline Grey** (#C6CDD1): Rules and input borders.
- **Night Plate** (#0D1215): Dark ground.
- **Night Ink** (#E7ECEB): Dark body copy.
- **Night Hairline** (#2B373D): Dark rules.

### Named Rules
**The Single Accent Rule.** Cobalt is used only for active selection, links, and actions. If a region needs emphasis, use weight, scale, or a hairline instead.

**The Stamp Rule.** Status is a rotated square mark plus its text name. The mark is filled only for `available`; shape and text carry the meaning without relying on hue.

## Typography

**Display Font:** Archivo (with ui-sans-serif fallback)
**Body Font:** Archivo
**Label/Mono Font:** IBM Plex Mono

**Character:** Archivo has the stolid, industrial proportions of a parts catalogue. IBM Plex Mono supplies measurement and evidence: indices, counts, config fragments, and the setup prompt.

### Hierarchy
- **Display** (500, clamp(2.4rem, 4.6vw, 4.1rem), 1.02): Catalog and entry titles, tracking -0.04em.
- **Title** (500, 1.32rem, 1.2): Entry names in the list.
- **Body** (400, 0.98rem, 1.6): Copy, capped near 65ch.
- **Label** (400, 0.72rem, 1.5, uppercase): Mono labels, counts, status, metadata.

### Named Rules
**The Tabular Rule.** Every numeric value uses tabular figures so columns line up without alignment tricks.

## Layout

Two columns at desktop: 232px spine and a fluid reading column. The spine is sticky at 100dvh; the main column holds either the catalog list or an entry article capped near 84ch. The list is a single column of ruled rows; there is no card grid and no right rail. Below 760px the spine becomes a top bar and its type filters wrap into a chip row. Gutter scales from 24px to 56px. Section spacing favors more room above a heading than below it.

## Elevation & Depth

No shadows. Depth comes from surface steps (Cool Paper to Plate White), hairline rules, and the sticky rail boundaries. Hover moves a row 6px rather than lifting it.

## Shapes

Corners are 2px everywhere; pills are reserved for mobile filter chips. The recurring geometry is the registration cross at the spine head and the rotated square status stamp. No gradients and no background textures.

## Components

### Buttons
- **Shape:** 2px radius, min-height 46px for primary actions.
- **Primary:** Cobalt fill, Plate White label in light mode; Night Plate label in dark mode.
- **Hover / Focus:** Slight brightness change; 1px downward press on active; visible 2px focus ring offset 3px.

### Chips
- **Style:** Mono labels with counts; mobile uses a 1px hairline border.
- **State:** Active filter gets Cobalt text and a surface-2 fill.

### Cards / Containers
- **Corner Style:** 2px.
- **Background:** Plate White on Cool Paper.
- **Shadow Strategy:** None; hairline border only.
- **Internal Padding:** 14px to 28px depending on role.

### Inputs / Fields
- **Style:** 1px hairline border, Plate White fill, mono text.
- **Focus:** Border changes to Cobalt.

### Navigation
The spine holds the wordmark, totals, type filters, theme toggle, and contribution link. It becomes a stacked top bar below 760px.

### Manifest Strip
The signature component: a bordered mono block with line numbers, a scroll cap on desktop, and a copy action that confirms in place. Copy always uses the original unwrapped prompt.

## Do's and Don'ts

### Do:
- **Do** keep the index spine present from the first viewport on desktop and let the list use the full remaining width.
- **Do** use tabular mono figures for every count and index.
- **Do** respect `prefers-reduced-motion` and remove the row entrance entirely.

### Don't:
- **Don't** introduce a second accent color or a gradient.
- **Don't** replace list rows with cards or add decorative icons.
- **Don't** animate layout properties; use transform and opacity only.
