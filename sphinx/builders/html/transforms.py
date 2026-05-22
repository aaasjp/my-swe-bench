"""
    sphinx.builders.html.transforms
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Transforms for HTML builder.

    :copyright: Copyright 2007-2020 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re
from typing import Any, Dict, List

from docutils import nodes

from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.nodes import NodeMatcher


class KeyboardTransform(SphinxPostTransform):
    """Transform :kbd: role to more detailed form.

    Before::

        <literal class="kbd">
            Control-x

    After::

        <literal class="kbd">
            <literal class="kbd">
                Control
            -
            <literal class="kbd">
                x
    """
    default_priority = 400
    builders = ('html',)
    pattern = re.compile(r'(-|\+|\^|\s+)')

    def run(self, **kwargs: Any) -> None:
        matcher = NodeMatcher(nodes.literal, classes=["kbd"])
        for node in self.document.traverse(matcher):  # type: nodes.literal
            parts = self.pattern.split(node[-1].astext())
            if len(parts) == 1:
                continue

            node.pop()

            result = []
            i = 0
            while i < len(parts):
                if parts[i] == '':
                    i += 1
                    continue

                key = parts[i]
                i += 1

                if i < len(parts):
                    sep = parts[i]
                    i += 1
                else:
                    sep = ''

                if sep and i < len(parts) and parts[i] != '':
                    # Separator between two non-empty parts: treat as text
                    result.append(key)
                    result.append(sep)
                else:
                    # Separator at boundary or between empty parts: merge
                    if sep:
                        if result and isinstance(result[-1], str):
                            result[-1] += sep
                        elif result:
                            result[-1] += sep
                        else:
                            result.append(sep)
                    if key:
                        result.append(key)

            # Add remaining content
            if not result:
                result.append('')
            elif isinstance(result[-1], str) and result[-1] == '':
                result.pop()

            for item in result:
                node += nodes.literal('', item, classes=["kbd"])


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_post_transform(KeyboardTransform)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
