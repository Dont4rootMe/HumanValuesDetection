def merge_entities(text, entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        try:
            if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
                current['text'] = text[current['start']: next_entity['end']].strip()
                current['end'] = next_entity['end']
            else:
                merged.append(current)
                current = next_entity
        except: pass

    # Append the last entity
    merged.append(current)
    return merged