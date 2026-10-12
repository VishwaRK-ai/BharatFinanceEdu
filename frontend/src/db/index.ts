/**
 * Neon Postgres client (neon-http driver) — server-only.
 *
 * Never import this module from client components: `getServerEnv()` throws
 * when touched in the browser, and the DB connection string must never reach
 * the client bundle.
 */
import { neon } from "@neondatabase/serverless";
import { drizzle, type NeonHttpDatabase } from "drizzle-orm/neon-http";
import { getServerEnv } from "@/lib/env";
import * as schema from "./schema";

let _db: NeonHttpDatabase<typeof schema> | null = null;

export function getDb(): NeonHttpDatabase<typeof schema> {
  if (!_db) {
    const env = getServerEnv();
    const sql = neon(env.DATABASE_URL);
    _db = drizzle({ client: sql, schema });
  }
  return _db;
}

export const db = new Proxy({} as NeonHttpDatabase<typeof schema>, {
  get(_target, prop, receiver) {
    const instance = getDb();
    const value = Reflect.get(instance, prop, receiver);
    if (typeof value === "function") {
      return value.bind(instance);
    }
    return value;
  },
});

export type Database = NeonHttpDatabase<typeof schema>;
