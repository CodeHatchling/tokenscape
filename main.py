"""Dasher‑style viewer prototype with PyQt6 + PyGLM
Single root Node, arrow keys move the View (camera).
‑ Requires:  PyGLM  (pip install PyGLM)
"""

from __future__ import annotations
import math
import sys
import time
import unicodedata
from pyglm import glm
from dataclasses import dataclass
from PyQt6.QtWidgets import (
    QApplication, QWidget, QMainWindow,
    QTextEdit, QDockWidget, QStyleOptionButton,
    QVBoxLayout, QLabel, QFrame,
    QScrollArea, QMenu
)
from PyQt6.QtGui import QPainter, QPen, QTextCursor
from PyQt6.QtCore import (
    Qt, QRectF, QTimer,
    QPoint, QPointF, QSize,
    QSizeF, QRect
)

from concurrent.futures import Future

import llm_thread
import llm_wrapper
# from timer import Timer, timed

vec2 = glm.vec2


# ---------------------------- Math helpers ---------------------------
def softplus(x: float, softness: float):
    if x / softness > 30:
        return x
    else:
        return math.log(1 + math.exp(x / softness)) * softness


def softabs(x: float, softness: float):
    return softplus(x, softness) + softplus(-x, softness)


def softmax(x: float, y: float, softness: float, must_be_less: bool):
    # Result is the same when x and y are swapped!
    x_minus_y = x - y
    soft_abs = softabs(x_minus_y, softness)
    if must_be_less:
        soft_abs -= softabs(0, softness)
    return (x_minus_y + softabs(x_minus_y, softness)) / 2 + y


def softmin(x: float, y: float, softness: float, must_be_greater: bool):
    return -softmax(-x, -y, softness, must_be_greater)


# ---------------------------- LLM related ----------------------------
# Load model
model_path = input("Enter the path to the model file: ").strip(' \'\"\t\n\r')  # Remove quotes and whitespace
llm = llm_wrapper.get_llm(model_path=model_path, n_ctx=8192, n_gpu_layers=-1, logits_all=True)
thread = llm_thread.LLMThread()


# ---------------------------- Geometry helpers ----------------------------
@dataclass
class Rect:
    min: vec2
    max: vec2

    def __str__(self):
        return f"Rect({self.min.x}, {self.min.y}, {self.max.x}, {self.max.y})"

    @staticmethod
    def identity() -> Rect:
        return Rect(vec2(0, 0), vec2(1, 1))

    @property
    def size(self) -> vec2:
        return self.max - self.min

    @property
    def center(self) -> vec2:
        return vec2(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0
        )

    def overlaps(self, other: Rect) -> bool:
        # Returns true if this rect overlaps with the other rect in whole or in part.
        if self.max.x < other.min.x or self.min.x > other.max.x:
            return False
        if self.max.y < other.min.y or self.min.y > other.max.y:
            return False
        return True

    def encompasses(self, other: Rect) -> bool:
        # Returns true if this rect contains the other rect in its entirety;
        # i.e. if the other rect is entirely within this rect.
        return (
            other.max.x <= self.max.x and
            other.min.x >= self.min.x and
            other.max.y <= self.max.y and
            other.min.y >= self.min.y
        )

    def contains(self, point: vec2) -> bool:
        if self.max.x < point.x or self.min.x > point.x:
            return False
        if self.max.y < point.y or self.min.y > point.y:
            return False
        return True

    def clamp_point(self, point: vec2) -> vec2:
        return vec2(
            min(max(point.x, self.min.x), self.max.x),
            min(max(point.y, self.min.y), self.max.y)
        )

    def transform_point(self, point: vec2) -> vec2:
        return point * self.size + self.min

    def inverse_transform_point(self, point: vec2) -> vec2:
        return (point - self.min) / self.size

    def transform_rect(self, other: Rect) -> Rect:
        return Rect(self.transform_point(other.min), self.transform_point(other.max))

    def inverse_transform_rect(self, other: Rect) -> Rect:
        return Rect(self.inverse_transform_point(other.min), self.inverse_transform_point(other.max))

    def get_flip(self) -> tuple[bool, bool]:
        x_flip = self.min.x > self.max.x
        y_flip = self.min.y > self.max.y
        return x_flip, y_flip

    def flip(self, flip_x: bool, flip_y: bool) -> Rect:
        if flip_x:
            self.min.x, self.max.x = self.max.x, self.min.x
        if flip_y:
            self.min.y, self.max.y = self.max.y, self.min.y
        return self

    @property
    def unflipped(self) -> Rect:
        return Rect(vec2(min(self.min.x, self.max.x), min(self.min.y, self.max.y)),
                    vec2(max(self.min.x, self.max.x), max(self.min.y, self.max.y)))

    def clamp_rect(self, other: Rect) -> Rect:
        return Rect(self.clamp_point(other.min), self.clamp_point(other.max))

    def expand(self, pixels: vec2) -> Rect:
        # In the case of negative pixels, this will shrink the rect.
        # We need to limit the shrinkage to prevent its size from becoming negative.
        # This is a bit of a hack, but it works.

        minimum_contraction = vec2(
            max(self.size.x, 0.0) / 2.0,
            max(self.size.y, 0.0) / 2.0
        )

        pixels = vec2(
            max(-minimum_contraction.x, pixels.x),
            max(-minimum_contraction.y, pixels.y)
        )

        new_min = self.min - pixels
        new_max = self.max + pixels
        return Rect(new_min, new_max)

    def modified(self,
                 xmin: None | float = None,
                 xmax: None | float = None,
                 ymin: None | float = None,
                 ymax: None | float = None) -> Rect:

        return Rect(
            vec2(xmin if xmin is not None else self.min.x, ymin if ymin is not None else self.min.y),
            vec2(xmax if xmax is not None else self.max.x, ymax if ymax is not None else self.max.y)
        )


# ------------------------------- Model ------------------------------------
frame_count = 0


def range_to_local_rect(v0, v1) -> Rect:
    return Rect(
        vec2(0.0, v0),
        vec2(v1 - v0, v1)
    )


class Node:
    class ChildrenData:
        def __init__(self, parent_node: Node):
            # The tokens of the children of this node.
            self.token_values: list[int] = []

            # The children's token probabilities.
            self.probabilities: list[float] = []

            # The start and end of the range of the child in the parent node;
            # _accumulative_probabilities[n] = sum(_probabilities[:n])
            self.accumulative_probabilities: list[float] = []

            # The labels for the children (token n decoded by the model).
            self.token_strings: list[str] = []

            # We use a dictionary here because, in high-vocabulary models,
            # the vast majority of possible tokens will not even be visited by the view.
            self.child_nodes: dict[int, Node] = {}

            self.parent_node = parent_node

        @property
        def children(self):
            if self is None:
                return []
            return self.child_nodes.values()

        def get_child_index(self, vertical_position: float) -> int:
            # Using binary search, find the index of the child whose span contains the vertical position.

            # if the vertical position is too low, return the first child.
            if vertical_position < 0:
                return 0

            # if the vertical position is too high, return the last child.
            if vertical_position > 1:
                return len(self.accumulative_probabilities) - 1

            # Perform binary search:
            low = 0
            high = len(self.accumulative_probabilities) - 1
            while low <= high:
                mid = (low + high) // 2
                span_min = self.accumulative_probabilities[mid]
                span_max = (
                    self.accumulative_probabilities[mid + 1]
                    if mid + 1 < len(self.accumulative_probabilities)
                    else 1.0
                )
                if span_min <= vertical_position <= span_max:
                    return mid
                elif vertical_position < span_min:
                    high = mid - 1
                else:
                    low = mid + 1

            # If we reach here, the vertical position is not within any child's span.
            # But that's not possible, a value must have been found.
            # Let's return the last index.
            return len(self.accumulative_probabilities) - 1

        def get_child_start(self, index: int) -> float:
            if index < 0 or index >= len(self.accumulative_probabilities):
                raise IndexError("Child index out of range")
            return self.accumulative_probabilities[index]

        def get_child_end(self, index: int) -> float:
            if index < 0 or index >= len(self.accumulative_probabilities):
                raise IndexError("Child index out of range")
            return self.accumulative_probabilities[index + 1] if index + 1 < len(
                self.accumulative_probabilities) else 1.0

        def get_child_probability(self, index: int) -> float:
            if index < 0 or index >= len(self.probabilities):
                raise IndexError("Child index out of range")
            return self.probabilities[index]

        def get_child_token_string(self, index: int) -> str:
            if index < 0 or index >= len(self.token_strings):
                raise IndexError("Child index out of range")
            return self.token_strings[index]

        def get_child_node(self, index: int) -> Node:
            if index < 0 or index >= len(self.token_strings):
                raise IndexError("Child index out of range")

            if index in self.child_nodes:
                return self.child_nodes[index]

            # Create the child node and store it in the dictionary.
            child_token = self.token_values[index]
            child_token_str = self.token_strings[index]
            child_node = Node(
                vertical_span=(self.get_child_start(index), self.get_child_end(index)),
                parent=self.parent_node,
                token=child_token,
                token_str=child_token_str
            )
            self.child_nodes[index] = child_node
            return child_node

    label_size_pixels: vec2 | None = None
    label_string: str | None = None

    def __init__(self,
                 vertical_span=(0.0, 1.0),
                 parent=None,
                 token: int | list[int] | None = None,
                 token_str: str | None = None):

        # These values are intended to be immutable.
        self.parent: Node = parent
        self.v0, self.v1 = vertical_span
        self._token = token
        self.token_str: str | None = token_str

        # These values are evaluated lazily and accessed via method calls.
        self._token_sequence: tuple[int, ...] | None = None
        self._token_sequence_str: str | None = None
        self._children_data: Node.ChildrenData | None = None

        # These values are used to store the result of the threaded function call.
        self._future_children_data: Future | None = None

        self._last_frame_drawn = 0
        self._last_drawn_rect = Rect(vec2(0, 0), vec2(0, 0))

    @property
    def token_sequence_str(self) -> str:
        if self._token_sequence_str is not None:
            return self._token_sequence_str

        if self.parent is not None:
            # The field 'token_str' is assigned once when the Node is created.
            if self.token_str is not None:
                self._token_sequence_str = self.parent.token_sequence_str + self.token_str
            else:
                self._token_sequence_str = self.parent.token_sequence_str
        else:
            # The field 'token_str' is assigned once when the Node is created.
            if self.token_str is not None:
                self._token_sequence_str = self.token_str
            else:
                self._token_sequence_str = ""

        return self._token_sequence_str

    @property
    def token_sequence(self) -> tuple[int, ...]:
        if self._token_sequence is not None:
            return self._token_sequence

        # with Timer("Node.token_sequence"):
        # If the parent is None, we are the root node.
        if self.parent is not None:
            parent_seq = self.parent.token_sequence
        else:
            parent_seq = ()

        if isinstance(self._token, list):
            self._token_sequence = parent_seq + tuple(self._token)
        elif isinstance(self._token, int):
            self._token_sequence = parent_seq + (self._token,)
        else:
            self._token_sequence = parent_seq

        return self._token_sequence

    def _threaded_get_priority(self) -> (int, int):
        # This function is called in a separate thread,
        # however, it only reads data that is calculated on the main thread.
        # This means that we can safely access the data without locking.
        # We return how many frames ago we were drawn, and the size of the node when it was last drawn.
        return self._last_frame_drawn, self._last_drawn_rect.size.x * self._last_drawn_rect.size.y

    def _threaded_get_children_data(self, token_sequence: list[int], top_k: int) -> Node.ChildrenData:
        next_tokens, next_probs, _ = llm.query_tokens(token_sequence, top_k)

        # Sort the tokens alphabetically by their string representation.
        combined = zip(next_tokens, next_probs)
        sorted_tokens = sorted(combined, key=lambda x: -llm.token_to_alpha_order(x[0]), reverse=True)
        next_tokens, next_probs = zip(*sorted_tokens)

        # Since we may not be capturing the entire token vocabulary,
        # we need to ensure that the probabilities sum to 1.
        total_prob = 0
        for i in range(len(next_tokens)):
            total_prob += next_probs[i]

        children_data = Node.ChildrenData(self)

        prob_sum = 0
        for i in range(len(next_tokens)):
            child_probability = next_probs[i] / total_prob
            next_prob_sum = prob_sum + child_probability
            child_token = next_tokens[i]
            child_token_str = llm.tokens_to_str([child_token])

            children_data.token_values.append(child_token)
            children_data.probabilities.append(child_probability)
            children_data.accumulative_probabilities.append(prob_sum)
            children_data.token_strings.append(child_token_str)

            # We don't create the child node here, because we don't want to create
            # all the children at once, as this would be inefficient.
            # Instead, we create the child node when it is requested.

            prob_sum = next_prob_sum

        return children_data

    def get_children_data(self) -> Node.ChildrenData | None:
        if self._children_data is not None:
            return self._children_data

        if self._future_children_data is None:
            self._future_children_data = thread.submit(
                self._threaded_get_children_data,
                self._threaded_get_priority,
                self.token_sequence, 500)
            return None

        if not self._future_children_data.done():
            return None

        if self._future_children_data.exception() is not None:
            # Handle the exception here
            print(f"Exception in future: {self._future_children_data.exception()}")
            self._future_children_data = None
            return None

        # Get the result of the future
        self._children_data = self._future_children_data.result()
        return self._children_data

    @property
    def local_rect(self) -> Rect:
        return range_to_local_rect(self.v0, self.v1)

    def update_visibility_prioritization(self, rect: Rect):
        global frame_count
        self._last_frame_drawn = frame_count
        self._last_drawn_rect = rect


class View:
    """Defines camera position within its parent node's coordinate system"""

    def __init__(self, parent: Node, position: vec2 = vec2(0.25, 0.5)):
        self.parent = parent
        self.position: glm.vec2 = position  # vec2(x,y)

    @property
    def local_rect(self) -> Rect:
        x_min = 0.0
        x_max = self.position.x * 2.0
        half_w = (x_max - x_min) / 2.0
        y_min = self.position.y - half_w
        y_max = self.position.y + half_w
        return Rect(glm.vec2(x_min, y_min), vec2(x_max, y_max))


def rect_to_QRect(rect: Rect) -> QRectF:
    size = rect.size
    return QRectF(rect.min.x, rect.min.y, size.x, size.y)


drawing_style = QApplication.style()
drawing_option = QStyleOptionButton()


def transform_rect_for_drawing(pixel_rect: Rect, cull_rect: Rect) -> Rect:
    def flip(rect):
        # Flip horizontally within the viewport.
        return Rect(
            vec2(1 - rect.max.x, rect.min.y),
            vec2(1 - rect.min.x, rect.max.y)
        )

    pixel_rect = cull_rect.inverse_transform_rect(pixel_rect)
    pixel_rect = flip(pixel_rect)
    pixel_rect = cull_rect.transform_rect(pixel_rect)
    return pixel_rect




# This is a mapping of invisible characters (or characters that are not visible in isolation) to visible characters,
# excluding "Cf", "Zs", "Cc", "Mn" categories.
replacement_map = {
    ord(" "): "␣",
    ord("\t"): "⇥",
    ord("\n"): "↵",
    ord("\r"): "↵",
    ord("\f"): "␌",
    ord("\v"): "␊",
    ord("\a"): "␍",
    ord("\b"): "␈",
    0x00A0: "⍽",      # no-break space
    0x200B: "🕳️",     # zero-width space
    0x2009: "·",      # thin space
    0x202F: "⍽",      # narrow no-break space
    0x00AD: "-",      # soft hyphen (your call)
    0x2800: '␣',      # braille blank
    0x200C: "␠",     # zero-width non-joiner
    0x200D: "␠",     # zero-width joiner
    0x2060: "␠",     # word joiner
    0xFEFF: "␠",     # zero-width non-breaking space
    0x2028: "␠",     # line separator
    0x2029: "␠",     # paragraph separator
    0x2063: "␠",     # invisible separator
    0x2064: "␠",     # invisible times
    0x2065: "␠",     # invisible separator
    0x2066: "␠",     # left-to-right embedding
    0x2067: "␠",     # left-to-right override
    0x2068: "␠",     # right-to-left embedding
}


def replace_invisible_char(c: str) -> str:
    cp = ord(c)
    if cp in replacement_map:
        return replacement_map[cp]

    try:
        name = unicodedata.name(c)
        acronym = ''.join(word[0] for word in name.split()).upper()
        return f"␠{acronym}"
    except ValueError:
        return f"[{cp:X}]"


def is_char_invisible(c: str) -> bool:
    try:
        if c.isspace():
            return True

        category = unicodedata.category(c)
        if category in {"Cf", "Zs", "Cc", "Mn"}:  # Format, Separator-Space, Control, Nonspacing
            return True

        return not c.isprintable()
    except Exception:
        return False


def replace_whitespace_only_labels(label: str) -> str:
    for char in label:
        if not is_char_invisible(char):
            return label

    # If we have a whitespace-only or invisible-only label, we need to replace the characters with visible counterparts.
    replaced_string: str = ""
    for char in label:
        if is_char_invisible(char):
            replaced_string += replace_invisible_char(char)
        else:
            replaced_string += char

    return replaced_string


cache: dict[str, tuple[str, vec2]] = {}


def get_label_info(text: str, painter: QPainter) -> tuple[str, vec2]:
    global cache

    if text in cache:
        return cache[text]

    metrics = painter.fontMetrics()

    label = text
    label = replace_whitespace_only_labels(label)

    size_qsize = metrics.size(0, text)

    # If the string is too long, truncate it. Also, what the hell is this comment?
    if size_qsize.width() > 100:
        # Truncate the string.
        label = label[:20] + "..."

    size_qsize = metrics.size(0, label)
    size = QSize_to_vec2(size_qsize)
    result = (label, size)
    cache[text] = result

    return result

def get_view_clamped_rect(rect: Rect, cull_rect: Rect, label_width: float):
    # Clamp drawn rect to cull rect while leaving room for the label offscreen.
    clamper_rect = Rect(
        vec2(cull_rect.min.x, cull_rect.min.y),
        vec2(cull_rect.max.x + label_width, cull_rect.max.y)
    )
    return clamper_rect.clamp_rect(rect)


def get_drawing_info(pixel_rect: Rect, cull_rect: Rect, text: str, min_height: int, painter: QPainter) \
        -> tuple[QRect, str, vec2] | None:
    global cache

    # Not visible?
    if not cull_rect.overlaps(pixel_rect):
        return None

    string_to_write, size = get_label_info(text, painter)

    # Clamp drawn rect to cull rect while leaving room for the label offscreen.
    clamped_rect = get_view_clamped_rect(pixel_rect, cull_rect, size.x)
    clamped_size = clamped_rect.size

    # Too small?
    if clamped_size.x < size.x or clamped_size.y < min_height:
        return None

    return rect_to_QRect(transform_rect_for_drawing(clamped_rect, cull_rect)).toRect(), string_to_write, size


def draw_background(painter: QPainter, q_rect: QRect):
    painter.setOpacity(0.25)
    painter.setPen(QPen(drawing_style.standardPalette().accent().color(), 1))
    painter.drawLine(QPoint(q_rect.x(), q_rect.y()), QPoint(q_rect.x(), q_rect.y() + q_rect.height()))

    painter.fillRect(q_rect, drawing_style.standardPalette().window())


def draw_text(painter: QPainter, q_rect: QRect, label: str):
    painter.setOpacity(1)
    painter.setPen(QPen(drawing_style.standardPalette().windowText().color(), 1))
    painter.drawText(q_rect,
                     Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter,
                     label)


def draw_loading_text(painter: QPainter, q_rect: QRect, label: str):
    painter.setOpacity(0.5)
    painter.setPen(QPen(drawing_style.standardPalette().windowText().color(), 1))

    font = painter.font()

    # Store the current font size
    point_size = font.pointSize()

    # Set the font size to 20 for the loading text
    font.setPointSize(20)
    painter.setFont(font)

    painter.drawText(q_rect,
                     Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter,
                     label)

    # Reset the font size to the original
    font.setPointSize(point_size)
    painter.setFont(font)


loading_gizmo_frames = [
    "◜",
    "◠",
    "◝",
    "◞",
    "◡",
    "◟",
]


def get_loading_text(frames, speed=10) -> str:
    i = int(time.time() * speed) % len(frames)
    return frames[i]


def draw_node(node: Node, pixel_rect: Rect, cull_rect: Rect, min_height: int, painter: QPainter) -> bool:
    def expand_rect(_child_rect: Rect) -> Rect:
        # As the rect's "base" size grows, we will smoothly interpolate from the expanded size to the base size.
        # This is to prevent sudden jumps when the view is reparented, by restricting the influence of scaling
        # to content within the cull rect.
        ratio = _child_rect.size.x / cull_rect.size.x
        if ratio > 1.0:
            return _child_rect

        parent_limit = pixel_rect.max.x
        child_limit = _child_rect.max.x
        desired_x = (_child_rect.max.x + cull_rect.max.x) / 2.0
        limited_x = softmax(desired_x, child_limit, 1, False)
        limited_x = softmin(limited_x, parent_limit, 50, False)
        _child_rect = _child_rect.modified(
            xmax=limited_x * (1-ratio) + _child_rect.max.x * ratio,
        )
        return _child_rect

    # Get the drawing stuff for this node.
    drawing_info = get_drawing_info(pixel_rect, cull_rect, node.token_str, min_height, painter)
    # If None is returned, it means the node is not visible, and we can skip drawing it.
    if drawing_info is None:
        return False
    q_rect, label_string, label_size = drawing_info

    draw_background(painter, q_rect)

    original_pixel_rect = pixel_rect

    # Retract the rect to make room for the label.
    # BUG: This causes sudden jumps when the view is reparented.
    pixel_rect = Rect(
        pixel_rect.min,
        vec2(max(pixel_rect.min.x, pixel_rect.max.x - label_size.x), pixel_rect.max.y),
    )

    # Find the visible vertical span (range 0 to 1) of the node,
    # which we will use to figure out what children to draw.
    pixel_rect_height = pixel_rect.max.y - pixel_rect.min.y
    start_range = (cull_rect.min.y - pixel_rect.min.y) / pixel_rect_height
    end_range = (cull_rect.max.y - pixel_rect.min.y) / pixel_rect_height
    start_range = max(0.0, min(1.0, start_range))
    end_range = max(0.0, min(1.0, end_range))
    visible_range = (start_range, end_range)

    child_data = node.get_children_data()
    greatest_child_max_x = pixel_rect.min.x

    # If the children data is available, we can draw the children.
    if child_data is not None:
        # Find the starting and ending indices of the children to draw.
        start_index = child_data.get_child_index(visible_range[0])
        end_index = child_data.get_child_index(visible_range[1])

        # The minimum accumulative probability required to draw a child/group of children.
        probability_required = min_height / pixel_rect_height

        # Since many of the children will likely be too small to draw,
        # we will draw a rectangle for "groups" of children whose heights sum to the minimum height,
        # and draw the highest scoring child to represent the group using the group rect.
        # This process may only yield a single child,
        # in which case we will draw it directly and not use the group rect.

        probability_start = child_data.get_child_start(start_index)
        probability_accumulative = 0.0
        highest_child_index = -1
        highest_child_probability = -1e10
        child_count = 0
        for i in range(start_index, end_index + 1):
            child_probability = child_data.get_child_probability(i)
            probability_accumulative += child_probability

            if child_probability > highest_child_probability:
                highest_child_index = i
                highest_child_probability = child_probability

            child_count += 1

            if probability_accumulative < probability_required:
                # We don't have enough probability to draw this child.

                # BUT, if the next child doesn't exist, we should draw it despite it not being large enough.
                # ALSO, Check if the next child is large enough to draw on its own.
                # In this case we should draw this group despite it not being large enough as well,
                # as to avoid interfering with the next child.
                if (i + 1) <= end_index and child_data.get_child_probability(i + 1) < probability_required:
                    continue

            if child_count == 1:
                # Draw the child directly
                # This call will create the children of the node, if they don't exist yet.
                child = child_data.get_child_node(i)
                child_rect = pixel_rect.transform_rect(child.local_rect)
                child_rect = expand_rect(child_rect)
                if draw_node(child, child_rect, cull_rect, min_height, painter):
                    if greatest_child_max_x < child_rect.max.x:
                        greatest_child_max_x = child_rect.max.x
            else:
                best_child_node = child_data.get_child_node(highest_child_index)

                # Draw a rectangle for the group of children
                group_rect = range_to_local_rect(probability_start, probability_start + probability_accumulative)
                group_rect = pixel_rect.transform_rect(group_rect)
                group_rect = expand_rect(group_rect)
                if draw_node(best_child_node, group_rect, cull_rect, min_height, painter):
                    if greatest_child_max_x < group_rect.max.x:
                        greatest_child_max_x = group_rect.max.x

            probability_start = child_data.get_child_end(i)
            probability_accumulative = 0.0
            highest_child_index = -1
            highest_child_probability = -1e10
            child_count = 0

    # Otherwise, the children data is not available yet.
    else:
        # Update the node's priority, which is based on how recently it the last draw attempt was,
        # and the size of the node when it was last drawn.
        node.update_visibility_prioritization(pixel_rect)

        # Draw a loading indicator.
        global loading_gizmo_frames

        group_drawing_info = get_drawing_info(
            pixel_rect,
            cull_rect,
            get_loading_text(loading_gizmo_frames),
            min_height,
            painter)

        if group_drawing_info is not None:
            group_q_rect, group_label_string, group_label_size = group_drawing_info
            # Draw the rect and label for the group of children
            draw_background(painter, group_q_rect)
            draw_loading_text(painter, group_q_rect, group_label_string)

        if greatest_child_max_x < pixel_rect.max.x:
            greatest_child_max_x = pixel_rect.max.x

    # We draw the text after the rectangles (including children rectangles),
    # so that it is on top of them.
    # To draw the text, we find the empty space between the parent's rectangle and the largest child rectangle.
    # Then we draw the text centered within that space.
    text_rect = Rect(
        vec2(greatest_child_max_x, original_pixel_rect.min.y),
        original_pixel_rect.max
    )
    text_rect = get_view_clamped_rect(text_rect, cull_rect, label_size.x)
    q_text_rect = rect_to_QRect(transform_rect_for_drawing(text_rect, cull_rect)).toRect()
    draw_text(painter, q_text_rect, label_string)

    return True


# @timed("draw_view")
def draw_view(view: View, pixel_rect: Rect, painter: QPainter):
    parent = view.parent

    # get the parent rect in pixel space
    parent_rect_px = pixel_rect.transform_rect(view.local_rect.inverse_transform_rect(Rect.identity()))

    draw_node(parent, parent_rect_px, pixel_rect, 15, painter)


# @timed("update_view_parenting")
def update_view_parenting(view: View):
    # Note: The view's local rect and the parent's local rect do not share the same coordinate system;
    #       Instead, the view is described within the parent's local rect,
    #       and the parent's local rect is described within its own parent's rect.
    #       Thus checking if the parent rect contains the view rect
    #       just means comparing it to an identity rect.

    # If the parent rect does not entirely encompass the view rect, reparent upwards.
    while not Rect.identity().encompasses(view.local_rect):
        if view.parent.parent is None:
            break

        # Describe the view within the new parent rect
        view.position = view.parent.local_rect.transform_point(view.position)
        # Reparent to the new parent node
        view.parent = view.parent.parent

    # NOTE: The view rect and a child rect are in the same coordinate system,
    #       because they are both parented to the same parent.
    #       Thus checking if the child rect contains the view rect
    #       is just a matter of comparing their local rects.

    # If a child rect entirely contains the view rect, reparent downwards.
    child_data = view.parent.get_children_data()
    if child_data is not None:
        for child in child_data.children:
            child_rect = child.local_rect
            if child_rect.encompasses(view.local_rect):
                # Describe the view within the new parent rect (a child of its current parent)
                view.position = child_rect.inverse_transform_point(view.position)
                # Reparent
                view.parent = child
                return


def integrated_move(pos: vec2, vel: vec2, dt: float):
    a = vel.x
    x_new = pos.x * math.exp(a * dt)

    if abs(a) < 1e-12:  # handle a ≈ 0 safely
        factor = dt * pos.x
    else:
        factor = (x_new - pos.x) / a  # (e^{a t}-1)/a * x0

    return vec2(
        x_new,
        pos.y + vel.y * factor
    )


def lerp(a, b, t):
    return a * (1-t) + b * t


def QPoint_to_vec2(point: QPoint | QPointF) -> vec2:
    return vec2(point.x(), point.y())


def QSize_to_vec2(size: QSize | QSizeF) -> vec2:
    return vec2(size.width(), size.height())


# --------------------------- Rendering Widget -----------------------------
class Canvas(QWidget):
    view: View

    def __init__(self, edit: QTextEdit, file):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.edit = edit
        self.file = file

        # This is a flag to handle changes invoked by LLM navigation instead of user typing.
        self._is_automatic_text_change = False

        # Add am event handler for when the text is modified by the user.
        def on_text_changed():
            if self._is_automatic_text_change:
                return

            # Get user input and tokenize it.
            user_text = edit.toPlainText()
            user_tokens = llm.str_to_tokens(user_text, add_bos=True)
            root_node = Node(vertical_span=(0.0, 1.0), token=user_tokens, token_str=user_text)  # single root box

            # initial view pos
            self.view = View(root_node, glm.vec2(0.5, 0.5))

        # Initialize with the empty text field.
        on_text_changed()

        # Connect the text edit's textChanged signal to the on_text_changed function
        edit.textChanged.connect(on_text_changed)

        # Set up the timer for updating the view
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_all)
        self.timer.start(16)

        # Set up the view velocity and mouse possession
        self.view_velocity: vec2 = vec2(0, 0)
        self.target_view_velocity: vec2 = vec2(0, 0)
        self.mouse_possessed = False
        self.last_mouse_pos = None
        self.target_view_velocity_mouse: vec2 = vec2(0, 0)

    def update_all(self):
        try:
            dt = 16/1000
            self.view_velocity = lerp(self.target_view_velocity + self.target_view_velocity_mouse,
                                      self.view_velocity,
                                      0.01**dt)
            self.view.position = integrated_move(self.view.position, self.view_velocity * vec2(1, -1), dt)

            old_parent = self.view.parent
            update_view_parenting(self.view)

            if old_parent != self.view.parent:
                try:
                    # If the parent has changed, update the text edit with the new token sequence
                    self._is_automatic_text_change = True
                    self.edit.setText(self.view.parent.token_sequence_str)

                    # Write the token sequence to the file and save it
                    try:
                        # Clear the contents of the file.
                        self.file.seek(0)
                        self.file.truncate()
                        self.file.write(self.view.parent.token_sequence_str + "\n")
                        self.file.flush()
                    except Exception as e:
                        print(f"Error writing to file: {e}")

                finally:
                    self._is_automatic_text_change = False

                # Scroll the text edit to the bottom
                self.edit.moveCursor(QTextCursor.MoveOperation.End)

            global frame_count
            frame_count += 1
            self.update()
        except Exception:
            traceback.print_exc()

    # ----------------- Rendering -----------------
    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # painter.fillRect(self.rect(), QColor("white"))

            rect = Rect(
                vec2(0, 0),
                QSize_to_vec2(self.rect().size()),
            )

            draw_view(self.view, rect, painter)
        except Exception:
            traceback.print_exc()

    # ----------------- Keyboard navigation -----------------
    def keyPressEvent(self, event):
        try:
            key = event.key()
            move_step = 1

            if key == Qt.Key.Key_Left:
                self.target_view_velocity.x += move_step
            elif key == Qt.Key.Key_Right:
                self.target_view_velocity.x -= move_step

            if key == Qt.Key.Key_Up:
                self.target_view_velocity.y += move_step
            elif key == Qt.Key.Key_Down:
                self.target_view_velocity.y -= move_step
        except Exception:
            traceback.print_exc()

    def keyReleaseEvent(self, event):
        try:
            key = event.key()
            move_step = -1

            if key == Qt.Key.Key_Left:
                self.target_view_velocity.x += move_step
            elif key == Qt.Key.Key_Right:
                self.target_view_velocity.x -= move_step

            if key == Qt.Key.Key_Up:
                self.target_view_velocity.y += move_step
            elif key == Qt.Key.Key_Down:
                self.target_view_velocity.y -= move_step
        except Exception:
            traceback.print_exc()

    # ----------------- Mouse navigation -----------------
    def mousePressEvent(self, event):
        try:
            if event.button() == Qt.MouseButton.LeftButton:
                self.mouse_possessed = True
                # self.setCursor(Qt.CursorShape.BlankCursor)  # Optional: hide cursor
                self.last_mouse_pos = event.position().toPoint()
        except Exception:
            traceback.print_exc()

    def mouseMoveEvent(self, event):
        try:
            if self.mouse_possessed:
                current_pos = QPoint_to_vec2(event.position().toPoint())
                center = QPoint_to_vec2(self.rect().center())
                self.target_view_velocity_mouse = (current_pos - center) * -8.0 / QSize_to_vec2(self.rect().size())
        except Exception:
            traceback.print_exc()

    def mouseReleaseEvent(self, event):
        try:
            if event.button() == Qt.MouseButton.LeftButton:
                self.target_view_velocity_mouse = 0
                self.mouse_possessed = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
        except Exception:
            traceback.print_exc()


# ------------------------------- main --------------------------------------
if __name__ == "__main__":
    import traceback

    app = QApplication(sys.argv)

    # Set the application style
    # Here are all the app styles available in Qt:
    # ['Breeze', 'Oxygen', 'QtCurve', 'Windows', 'Fusion']
    # app.setStyle("Breeze")

    # Create and open text file for the decoded output with a unique name.
    unique_name_suffix = time.strftime("%Y%m%d-%H%M%S")
    text_file = open(f"text_{unique_name_suffix}.txt", "w")

    text = QTextEdit("")
    text.setMinimumHeight(25)

    drawing_style = app.style()

    win = QMainWindow()
    canvas = Canvas(text, text_file)
    win.setCentralWidget(canvas)

    # Create a dock widget for the text edit
    dock = QDockWidget()
    dock.setWidget(text)
    dock.setWindowTitle("Text Edit")
    dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

    # Add the dock widget to the main window
    win.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    # Create a dock widget to display controls and help information.
    help_dock = QDockWidget()
    help_dock.setWindowTitle("Help")
    help_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

    # Create a scroll area for the help dock widget
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QFrame.Shape.NoFrame)

    # Display the controls and help information in the help dock.
    help_layout = QVBoxLayout()
    help_label = QLabel(
        "<h3>Continuation Viewport</h3>"
        "<center>"
        "<table>"
        "<tr><td align=center colspan=3><strong>Controls</strong></td></tr>"
        "<tr><td align=right>⬆️⬇️</td><td>  :  </td><td>Move the view up and down.</td></tr>"
        "<tr><td align=right>➡️⬅️</td><td>  :  </td><td>Move the view inward or outward.</td></tr>"
        "<tr><td align=right>Click+Drag 🖱️</td><td>  :  </td><td>Move the view towards the mouse position.</td></tr>"
        "</table>"
        "</center>"
        "<p align=\"justify\">"
        "Text continuations are arranged alphabetically, from top to bottom, within a panel. "
        "Each panel represents a token: a word or word fragment. "
        "Within that panel are more token panels. And those token panels contain more still. "
        "These panels are arranged in a tree structure, showing multiple ways to continue the text."
        "</p>"
        "<p align=\"justify\">"
        "<strong>To write</strong> using the viewing plane, "
        "<strong>click and hold down the mouse button</strong> towards the desired text. "
        "This will 'zoom in' on that text, showing variations and longer continuations of it. "
        "If you are looking for a <strong>specific continuation</strong>, "
        "you can <strong>zoom in</strong> on where it should be alphabetically,"
        "or type it manually. "
        "<strong>If you make a mistake</strong>, you can 'back out' of a continuation "
        "by <strong>moving the view back to the left</strong>."
        "</p>"
        "<p align=\"justify\">"
        "You can also use the <strong>arrow keys</strong> to <strong>move the view</strong>."
        "</p>"
        "<p align=\"justify\">"
        "<strong>To reset the view</strong>, simply <strong>type in the text edit</strong> dock."
        "</p>"
        "<h3>Manual Text Editing</h3>"
        "<p align=\"justify\">"
        "You can <strong>type in the text edit</strong> dock. "
        "When the <strong>text is changed</strong>, "
        "the <strong>view will update</strong> to show the current token sequence."
        "</p>"
        "<h3>Saving</h3>"
        "<p align=\"justify\">"
        "The file is <strong>automatically saved</strong> to the same directory as the script. "
        "Each text file is given a <strong>unique filename</strong>."
        "</p>"
        "<h3>UI Docks</h3>"
        "<p align=\"justify\">"
        "You can <strong>re-open closed docks</strong> using the <strong>\"View\" menu.</strong>"
        "</p>"
    )
    help_label.setWordWrap(True)
    help_layout.addWidget(help_label)

    # Set the layout for the help dock widget
    help_widget = QWidget()
    help_widget.setLayout(help_layout)
    scroll_area.setWidget(help_widget)
    help_dock.setWidget(scroll_area)

    # Add the help dock widget to the main window
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, help_dock)

    # Get the menu that is created when you right-click on the dock area.
    dock_menu = win.findChild(QMenu)

    # Add a menu bar with a "View" menu to re-open closed docks.
    menu_bar = win.menuBar()
    view_menu = menu_bar.addMenu("View")
    view_menu.addAction(help_dock.toggleViewAction())
    view_menu.addAction(dock.toggleViewAction())

    # Set the main window's properties
    win.setWindowTitle("Tokenscape")
    win.resize(1000, 800)
    win.show()

    result = app.exec()
    text_file.close()
    sys.exit(result)
