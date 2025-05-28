# tokenscape
An interactive LLM-powered writing interface that offers a navigable visual map of possible continuations, enabling users to collaboratively steer the writing process.

**Overview**

Inspired by Dasher, Tokenscape is a writing tool that leverages large language models (LLMs) to generate a tree-like structure of possible sentence continuations, visualized as nested rectangles containing 'tokens': words, word fragments, punctuation, or other commonly encountered text patterns. With fluid and intuitive control, users can explore different writing paths in parallel, with the model magnifying the most coherent and relevant options.

This approach enables a more collaborative means of leveraging LLMs' capabilities without sacrificing creative control. Users can:

1. **Brainstorm**: Quickly explore multiple writing directions and have a pool of ideas to consider. Uncover unexpected connections and wordplay opportunities that might have been missed otherwise.
2. **Broaden Writing Styles**: Expand their writing repertoire by sampling from diverse linguistic patterns and sentence structures, which can help refine their own writing style.
3. **Break Writer's Block**: When faced with a creative impasse, the model will always have more ideas to offer, which can help spark new inspiration or help push through a difficult or uninteresting passage.
4. **Focus**: Apply their creative energy on the most important decisions, while easing the mental burden of less impactful yet still necessary writing routines.
5. **Stay in Control**: The model remains a tool, not a replacement. Users retain full agency over the writing process, and can engage with the model's suggestions to whatever degree they see fit, explore paths that diverge from the model's predictions, or manually write and edit the text independently.

**Key Features**

1. **Navigable Probability Tree**: Users can seamlessly explore the LLM's predictions through a continuous zooming and scrolling interface. Continuations are arranged alphabetically, and by zooming in on a specific path, the model will generate broader variations and deeper continuations of that passage. 
2. **LLM Integration**: Built on Llama_cpp, users can provide custom LLMs or utilize pre-trained text models that cater to specific writing needs.
3. **Real-time Parallel Generation**: The model simultaneously generates multiple writing paths in the background, focussing computational resources where the user is exploring.
4. **Instruction Model Support**: If you use the conversation format of the model, you can utilize user-assistant conversation style prompting techniques.

**Getting Started**

1. Clone the repository: `https://github.com/CodeHatchling/tokenscape.git` 
2. Install dependencies: `pip install -r requirements.txt` (On some systems, extra steps may be required for CUDA support.)
3. Download a GGUF model: You can use any pre-trained model in the GGUF format.
4.  Run the application: `python main.py` This will launch a console that asks for the model file path. (If the file could not be opened by llama_cpp, a really dumb testing model will be used that only outputs fruit names.)
5.  Start writing: Begin by typing in the input field or navigating the probability tree. Enjoy surfing the waves of possibilities!

**Future Development**

1. File save/load support: Currently the outputs are automatically saved into a uniquely named text file within the python script's directory.
2. Customization: Load custom style sheets, change navigation speed, layout preferences, font and color styles, etc.
3. Text Inpainting: By selecting a region of text, the user can navigate a tree of continuations conditioned on seamlessly connecting with the following text.
4. Sampling Settings: Expose the top-N parameter (currently set to 500), use techniques to filter unwanted continuations, such as those with repetitions, etc.
5. Keyboard-Only Navigation: Provide a way to quickly select continuations without leaving the keyboard, e.g. using [Tab] or [Control]+[WASD] to select nodes. More ideas welcome.
6. Spiritual Successor to Dasher: While this project is mainly intended as a creative tool, it could be expanded and adapted to assist users with physical limitations, such as those with motor impairments.

**Known Issues**

1. Some users reported issues with keyboard control on non-Windows platforms.
2. The color scheme used to render the tree is based on OS-dependent color schemes, which may not look as intended on all systems.
3. LLMs can sometimes behave unexpectedly or produce undesirable output. The visualization techniques expose even the most unlikely paths, which can lead to inappropriate or offensive suggestions despite post-training. Further, this can be used to intentionally bypass content guidelines. You are responsible for your choice of model and how you use it.

**Feedback and Contributions**

Your feedback and contributions are always welcome! The project is still in its early stages, and there are many opportunities for improvement and expansion. Please feel free to open an issue, create a pull request/fork, or participate in discussions on the GitHub page. If you have any questions or ideas, feel free to reach out to me directly. Stay creative and happy writing!

(Note: This README.MD was generated using Tokenscape itself in combination with llama 3.1! ;;3)
