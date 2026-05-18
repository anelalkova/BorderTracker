-- Add latitude/longitude to crossings table
ALTER TABLE crossings
    ADD COLUMN IF NOT EXISTS latitude  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;

UPDATE crossings SET latitude = 41.135060600365165, longitude = 22.549000750027545 WHERE name = 'bogorodica';
UPDATE crossings SET latitude = 42.13921377796023, longitude = 21.305741596231403 WHERE name = 'blace';
UPDATE crossings SET latitude = 42.235180081087194, longitude = 21.704862868164376 WHERE name = 'tabanovce';
UPDATE crossings SET latitude = 42.221126212330134, longitude = 22.459115712211727 WHERE name = 'deve_bair';
UPDATE crossings SET latitude = 41.09191585867863, longitude = 20.609634718160073 WHERE name = 'kafasan';
UPDATE crossings SET latitude = 40.92035263629112, longitude = 21.41811780211385 WHERE name = 'medzitlija';
