import xml.etree.ElementTree as ET
import re

def strip_namespace(tag):
    """Removes the namespace URI from an XML tag for easy comparison."""
    return tag.split('}', 1)[1] if tag.startswith('{') else tag

def get_namespace(tag):
    """Extracts the namespace URI of an XML tag if it exists."""
    match = re.match(r'\{(.*)\}', tag)
    return match.group(1) if match else ''

def remove_transform_shape(element):
    """
    Recursively finds and removes Transform and Shape child elements.
    We iterate backwards through the children to safely delete elements 
    without breaking the iteration sequence.
    """
    for child in reversed(list(element)):
        tag_name = strip_namespace(child.tag)
        
        # If the child is a Transform or Shape, remove it (and its entire subtree)
        if tag_name in ('Transform', 'Shape'):
            element.remove(child)
        else:
            # Otherwise, traverse deeper into this child
            remove_transform_shape(child)

def clean_hanim_x3d(input_file, output_file):
    # 1. Parse the XML (X3D) file
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing X3D file: {e}")
        return

    # 2. X3D files usually declare a namespace. Registering it prevents
    # ElementTree from improperly prefixing output tags with "ns0:".
    ns = get_namespace(root.tag)
    if ns:
        ET.register_namespace('', ns)

    # 3. Collect all HAnimSegment and HAnimSite elements first.
    # We collect them into a list rather than modifying the tree while iterating over it.
    targets = []
    for elem in root.iter():
        tag_name = strip_namespace(elem.tag)
        if tag_name in ('HAnimSegment', 'HAnimSite'):
            targets.append(elem)

    # 4. Remove Transform and Shape elements from each targeted element's subtree
    for target in targets:
        remove_transform_shape(target)

    # 5. Write out the resulting XML
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Successfully processed the file. Output saved to: {output_file}")

if __name__ == "__main__":
    # Specify your input and output file paths here
    INPUT_X3D = 'JinScaledV2L1LOA4Markers11eJoeDemo5NoSkin.x3d'

    OUTPUT_X3D = 'JinScaledV2L1LOA4NoShapesNoSkin.x3d'
    
    clean_hanim_x3d(INPUT_X3D, OUTPUT_X3D)
