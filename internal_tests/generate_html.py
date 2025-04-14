from jinja2 import Environment, FileSystemLoader
import json
from markupsafe import Markup
import context

def escapejs(value):
    """
    Échappe les caractères spéciaux pour JavaScript.
    """
    if value is None:
        return ''
    escaped = (
        value.replace('\\', '\\\\')
             .replace("'", "\\'")
             .replace('"', '\\"')
             .replace('\n', '\\n')
             .replace('\r', '\\r')
             .replace('\t', '\\t')
    )
    return Markup(escaped)


def generate_html(template_file, output_file, context):

    env = Environment(loader=FileSystemLoader(""))
    env.filters['escapejs'] = escapejs

    template = env.get_template(template_file)

    rendered_html = template.render(context)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print(f"HTML généré avec succès dans {output_file}")


with open("test.json", "r", encoding="utf-8") as json_file:
    drawflow_json = json.load(json_file)
    
template_file = "index.html"
output_file = "output2.html"
# drawflow_json = update_positions(drawflow_json, box_width=250, separation=200, initial_y=200)
context = {
    "drawflow_json": json.dumps(drawflow_json)
}

generate_html(template_file, output_file, context)
