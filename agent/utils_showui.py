from browser_env.actions import ActionTypes, Action
from PIL import Image
import io
import base64
import os
import ast
# class Action(TypedDict):
#     action_type: int
#     coords: npt.NDArray[np.float32]
#     element_role: int
#     element_name: str
#     text: list[int]
#     page_number: int
#     url: str
#     nth: int
#     element_id: str
#     direction: str
#     key_comb: str
#     pw_code: str
#     answer: str
#     raw_prediction: str 

class ShowUIPromptConstructor:
    def __init__(self):
        self.min_pixels = 256*28*28
        self.max_pixels = 1344*28*28
        self._NAV_SYSTEM = """You are an assistant trained to navigate the web screen. 
                    Given a task instruction, a screen observation, and an action history sequence, 
                    output the next action and wait for the next observation. 
                    Here is the action space:
                    1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
                    2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
                    3. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
                    4. `ENTER`: Enter operation, value and position are not applicable.
                    5. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
                    """

    def construct_prompt_for_showui(self, goal, screenshot, action_history):
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        action_list = []
        # start from 1 because the first action_history is "None"
        for i, x in enumerate(action_history[1:], start=1):
            action_list.append(f'Step{i}: {x}')
        action_str = "; ".join(map(str,action_list))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._NAV_SYSTEM},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                    {"type": "text", "text": f"Task: {goal}\n"},
                    *([{"type": "text", "text": action_str}] if action_str != '' else []),
                ],
            }
        ]

        return messages
    
    def extract_action(self, action):
        try:
            # get the first action
            action = ast.literal_eval(action.split(";")[0])
            if isinstance(action, tuple) or isinstance(action, list):
                action = action[0]
        except:
            print("Error in parsing the action: ", action)
            action = action
        return action
    
    def construct(self, trajectory, intent, video_summary, page_screenshot_img, images, meta_data):
        return self.construct_prompt_for_showui(intent, page_screenshot_img, meta_data.get('action_history',[]))
        

def parse_showui_output(action):
    # Suppose showui navigation mode: {'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}
    # Now only support the following 5 action types:
    # √ 1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
    # √ 2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
    # × 3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required. 
    # √ 4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
    # × 5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
    # √ 6. `ENTER`: Enter operation, value and position are not applicable.
    # √ 7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
    # × 8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
    # × 9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.
    
    # for actions parsed previously (like stop action)
    if action.get("action_type"):
        return action
    
    parsed_action = Action()
    # map action type
    parsed_action["action_type"] = action_type_mapping(action)
    
    # map action position
    if action.get('position') is not None and not isinstance(action.get('position')[0], list):
        parsed_action["coords"] = [action.get('position')[0] * 1280, action.get('position')[1] * 720]
        
    # map action value
    if action.get('value') is not None:
        parsed_action["text"] = action.get('value')
        
    # map extra information (not used in the current version)
    # page = env.page
    # if action.get('position') is not None:
    #     x, y = action['position']
    #     element_info = find_element_by_coordinates_from_page(page, x, y)
    #     element_tagName = get_som_element_tagName(element_info)
    #     parsed_action["element_name"] = element_tagName
    
    # map special action
    if action.get("action") and action.get("action").lower() == "enter":
        parsed_action["key_comb"] = "Enter"
    if action.get("action") and action.get("action").lower() == "scroll":
        parsed_action["direction"] = action.get("value") # "up" or "down"
        
    return parsed_action
    


def action_type_mapping(action):
    if action.get('action') is None:
        print("No action found in the action object.")
        return ActionTypes.NONE
    match action['action'].lower():
        case 'click':
            return ActionTypes.MOUSE_CLICK
        case 'input':
            return ActionTypes.INPUT
        case 'hover':
            return ActionTypes.MOUSE_HOVER
        case 'enter':
            return ActionTypes.KEY_PRESS
        case 'scroll':
            return ActionTypes.SCROLL
        case _:
            return ActionTypes.NONE
        
import csv

ConditionTagNameList = [
    'span',
    'td',
    'th',
    'tr',
    'li',
    'div',
    'label',
    'filter-chip'
]

def get_page_info(page) -> list[list[float]]:
    """JavaScript code to return bounding boxes and other metadata from HTML elements."""
    js_script = """
    (() => {
        const interactableSelectors = [
            'a[href]:not(:has(img))', 'a[href] img', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
            '[tabindex]:not([tabindex="-1"])', '[contenteditable="true"]', '[role="button"]', '[role="link"]',
            '[role="checkbox"]', '[role="menuitem"]', '[role="tab"]', '[draggable="true"]',
            '.btn', 'a[href="/notifications"]', 'a[href="/submit"]', '.fa.fa-star.is-rating-item', 'input[type="checkbox"]'

        ];

        const textSelectors = ['p', 'span', 'div:not(:has(*))', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article'];
        const modifiedTextSelectors = textSelectors.map(selector =>
            `:not(${interactableSelectors.join(', ')}):not(style) > ${selector}`
        );

        const combinedSelectors = [...interactableSelectors, ...modifiedTextSelectors];
        const elements = document.querySelectorAll(combinedSelectors.join(', '));

        const pixelRatio = window.devicePixelRatio;
        let csvContent = "ID,Element,Element_role,Element_type,href,Top,Right,Bottom,Left,Width,Height,Alt,Class,Id,TextContent,Interactable\\n";
        let counter = 1;

        elements.forEach(element => {
            const rect = element.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return;
            let altText = element.getAttribute('alt') || '';
            altText = altText.replace(/"/g, ''); // Escape double quotes in alt text
            const classList = element.className || '';
            const id = element.id || '';
            let textContent = element.textContent || '';
            textContent = textContent.replace(/"/g, ''); // Escape double quotes in textContent
            const element_role = element.getAttribute('role') || ''; // Empty if not present
            const element_type = element.getAttribute('type') || ''; // Empty if not present
            const href = element.getAttribute('href') || ''; // Empty if not present
            // Determine if the element is interactable
            const isInteractable = interactableSelectors.some(selector => element.matches(selector));

            const dataString = [
                counter, element.tagName, element_role, element_type, href,(rect.top + window.scrollY) * pixelRatio,
                (rect.right + window.scrollX) * pixelRatio, (rect.bottom + window.scrollY) * pixelRatio,
                (rect.left + window.scrollX) * pixelRatio, rect.width * pixelRatio, rect.height * pixelRatio,
                altText, classList, id, textContent, isInteractable
            ].map(value => `"${value}"`).join(",");

            csvContent += dataString + "\\n";
            counter++;
        });

        return csvContent;
    })();
    """
    # Save the bbox as a CSV
    page_info = page.evaluate(js_script)
    return page_info

def construct_selector(row: dict) -> str:
    """
    Construct a Playwright selector from the CSV row data, with optional text content filtering.

    Parameters:
        row (dict): A dictionary representing a row from the CSV.

    Returns:
        str: A Playwright selector string.
    """
    tag = row["Element"].lower()
    element_id = row["Id"].strip()
    class_list = row["Class"].strip()
    include_string = row.get("TextContent", "").strip()  # Get 'TextContent', default to empty string if not found

    # Start with the tag name
    selector = tag

    # Add ID if available
    if element_id:
        selector += f"#{element_id}"

    # Add classes if available
    if class_list:
        classes = ".".join(class_list.split())  # Join classes with "."
        selector += f".{classes}"

    # Add text content filter if available
    if include_string:
        selector += f':has-text("{include_string}")'

    return selector

def get_som_element_tagName(element_info) -> str:
    tag_name = element_info["Element"].lower()
    if tag_name == 'input':
        input_type = element_info.get('Element_type')
        if input_type == 'checkbox':
            return 'checkbox'
        elif input_type == 'radio':
            return 'radio'
        elif input_type == 'button':
            return 'button'
        else:
            return 'input'
    elif tag_name == 'select':
        return 'select'
    elif tag_name == 'optgroup':
        return 'optgroup'
    elif tag_name == 'textarea':
        return 'textarea'
    elif tag_name == 'option':
        return 'option'
    elif tag_name == 'datalist':
        return 'datalist'
    elif tag_name == 'button':
        return 'button'
    elif tag_name == 'a':
        return 'link'
    elif tag_name in ConditionTagNameList:
        role = element_info.get('Element_role')
        if not role:
            return 'unknown'
        elif role == 'button':
            return 'button'
        elif role == 'link':
            return 'link'
        elif role == 'menuitem':
            return 'link'
        elif role == 'textbox':
            return 'input'
        elif role == 'checkbox':
            return 'checkbox'
        elif role == 'radio':
            return 'radio'
        elif role == 'tab':
            return 'link'
        elif role == 'switch':
            return 'switch'
        elif role == 'option':
            return 'option'
        elif role == 'row':
            return 'row'
        elif role == 'search-box':
            return 'search-box'
        else:
            return 'unknown'
    else:
        return 'unknown'

def find_element_by_coordinates(page_info: str, x: float, y: float):
    """
    Find the ID and TextContent of the element that contains the given (x, y) coordinates.

    Parameters:
        page_info (str): The CSV data containing element bounding boxes and metadata.
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        dict: A dictionary with 'ID' and 'TextContent' of the matching element, or None if not found.
    """
    # Parse the CSV content
    csv_reader = csv.DictReader(page_info.splitlines())

    for row in csv_reader:
        # Extract bounding box details from the row
        if row['Interactable'] != "true":continue
        top = float(row["Top"])
        right = float(row["Right"])
        bottom = float(row["Bottom"])
        left = float(row["Left"])

        # Check if the (x, y) point is inside the bounding box
        if left <= x<= right and top <= y <= bottom:
            row.update({
                "x": x,
                "y": y
            })
            return row

    # Return x,y if no matching element is found
    return {
        "x": x,
        "y": y
    }

def find_element_by_coordinates_from_page(page, x, y):
    if x<=1 and y<=1: 
        x = int(x * 1280) # "width": 1280
        y = int(y * 720) # "height": 720
    elements_info = get_page_info(page) 
    return find_element_by_coordinates(elements_info, x, y)
