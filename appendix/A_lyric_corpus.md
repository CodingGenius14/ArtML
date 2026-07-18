# Appendix A — Test Corpus: Full Lyric Text

The following synthetic lyric samples were used throughout all experiments in
the paper. They were written to represent distinct emotional registers while
avoiding copyright complications. Each mirrors a well-established register in
commercial popular music (confessional ballad, high-valence pop, and
ambition-driven hip-hop respectively) without reproducing any existing lyrics.

## Sad / Introspective

```
[Verse 1]
When the night falls and shadows creep
I find myself lost in memories deep
Every word you said, every tear I cried
In this darkness I search for the light inside

[Pre-Chorus]
The weight of the world on my shoulders now
I keep falling but I don't know how

[Chorus]
I am drowning in the rain tonight
Holding onto a flickering light
All these ghosts they will not let me sleep
Every promise was a wound that runs deep

[Bridge]
But I rise again, I figure it out somehow
Through the fog of who I used to be
```

Classifier output: sadness 0.699, fear 0.198 (correct).

## Upbeat Pop

```
[Verse 1]
Dancing under neon lights tonight
Feel the rhythm in the air so bright
Living for the moment, feeling alive
Watch me shine and watch me thrive

[Pre-Chorus]
Hands up in the summer sky
Confetti falling, we don't ask why

[Chorus]
We are golden, we are free
Glitter on the streets, you and me
Sun on our faces, music in the air
Love is everywhere, love is everywhere

[Bridge]
Forever young, forever bright
Chasing colors through the night
```

Classifier output: joy 0.980 (correct, high confidence).

## Rap / Ambition

```
[Verse 1]
Yeah, hear the beat drop, never gonna stop
From the bottom to the top, climbing without a flop
Stacking up my dreams, breaking through the seams
Nothing's ever what it seems, but I'm making it real

[Pre-Chorus]
Watch the city light up when I roll through
King of the skyline and I told you

[Chorus]
I run this city, gold chain, gold rings
Crown on my head, hear the choir sing
Lambo in the rearview, eyes on the throne
Came up from nothing, now I stand alone

[Bridge]
Hunger in my soul, full control
This is how I roll, focused on the goal
```

Classifier output: fear 0.331 — **misclassified**. This is the register-shift
failure that motivates the paper. The distribution is flat across all seven
classes with no decisive signal, in contrast to the confident correct readings
above.
