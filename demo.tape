Output images/demo.gif

Set Shell "zsh"
Set FontSize 14
Set Width 1200
Set Height 700
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

Type "echo '🔍 ProvenanceKit — Model Provenance Detection'"
Enter
Sleep 1s

Type "echo '⚖️  Compare two models to check if they share provenance'"
Enter
Sleep 500ms

Type "provenancekit compare bigscience/bloom-560m bigscience/bloomz-560m"
Enter
Sleep 15s

Sleep 3s

Type "echo '🔎 Scan a model against the known base-model database'"
Enter
Sleep 500ms

Type "provenancekit scan bigscience/bloom-560m --timing"
Enter
Sleep 20s

Sleep 3s
